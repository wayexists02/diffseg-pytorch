import numpy as np
import torch, einops
from torch import nn
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline
from typing import Dict, Optional, Union, Iterable
from .simple_ptp_utils import set_attention_processors, SimpleAttentionStore


@torch.inference_mode()
def diffseg_aggregate_self_attention_map(attention_store):
    self_attention_map = []
    for key in attention_store.keys():
        if "self" in key:
            self_attention_map.extend([am.mean(dim=1) for am in attention_store[key]]) # average over head dimension

    self_attention_map_sizes = list(map(lambda map: map.size()[-1], self_attention_map))
    self_attention_map_sizes = np.unique(self_attention_map_sizes)

    # get maximum size
    max_size = max(self_attention_map_sizes)
    
    aggretated_self_attention_map = []
    Rs = [] # aggregation weight (proportional to spatial size)

    # resize self attention maps
    for size in self_attention_map_sizes:
        self_attention_map_in_size = list(filter(lambda map: map.size()[-1] == size, self_attention_map))
        self_attention_map_in_size = torch.stack(self_attention_map_in_size, dim=1)
        N = self_attention_map_in_size.size(1)
        self_attention_map_in_size = einops.rearrange(
            self_attention_map_in_size,
            "b n h1 w1 h2 w2 -> (b n) (h1 w1) h2 w2",
        )
        self_attention_map_in_size = F.interpolate(self_attention_map_in_size, size=(max_size, max_size), mode="bilinear", align_corners=False)

        self_attention_map_in_size = einops.rearrange(
            self_attention_map_in_size,
            "bn (h1 w1) h2 w2 -> bn (h2 w2) h1 w1",
            h1=size,
            w1=size
        )
        self_attention_map_in_size = F.interpolate(self_attention_map_in_size, size=(max_size, max_size), mode="nearest")

        self_attention_map_in_size = einops.rearrange(
            self_attention_map_in_size,
            "(b n) (h2 w2) h1 w1 -> b n h1 w1 h2 w2",
            h2=max_size,
            w2=max_size,
            n=N
        )
        
        aggretated_self_attention_map.append(self_attention_map_in_size)
        Rs.extend([size] * N)

    aggretated_self_attention_map = torch.cat(aggretated_self_attention_map, dim=1)
    Rs = torch.tensor(Rs).to(aggretated_self_attention_map.device, aggretated_self_attention_map.dtype)
    Rs = Rs / Rs.sum()
    Rs = Rs[None, :, None, None, None, None]
    aggretated_self_attention_map = torch.sum(aggretated_self_attention_map * Rs, dim=1)
    aggretated_self_attention_map = aggretated_self_attention_map / aggretated_self_attention_map.sum(dim=(-1, -2), keepdim=True)
    return aggretated_self_attention_map


@torch.inference_mode()
def diffseg_compute_attention_distance(attention_map1: torch.Tensor, attention_map2: torch.Tensor):
    """
    Compute distance of self attention maps, as described in the paper (eq. 5).
    This function takes n pairs of self attention maps, and computes distance D for each pair.

    Arguments:
    - attention_map1: (n h w)
    - attention_map2: (n h w)

    Returns:
    - distances: (n,)
    """

    log_attention_map1 = attention_map1.log()
    log_attention_map2 = attention_map2.log()
    
    forward_kl = torch.sum(attention_map1 * (log_attention_map1 - log_attention_map2), dim=(-1, -2))
    reverse_kl = torch.sum(attention_map2 * (log_attention_map2 - log_attention_map1), dim=(-1, -2))
    return (forward_kl + reverse_kl)/2


@torch.inference_mode()
def diffseg_initialize_anchor_grid(anchor_grid_size: Iterable[int]):
    """
    Arguments:
    - anchor_grid_size: tuple of int (height, width)

    Returns:
    - anchor_grid: coordinates of each anchor point (1, height_of_anchor_grid, width_of_anchor_grid, 2)
    """

    # assume align_corners=True when grid sampling (coordinates of each corner point is [-1, -1], [-1, 1], [1, -1], [1, 1])
    anchor_x_coords = torch.linspace(-1, 1, steps=anchor_grid_size[1]*2 + 1)
    anchor_x_coords = anchor_x_coords[1:-1:2][None, :].expand(*anchor_grid_size)
    anchor_y_coords = torch.linspace(-1, 1, steps=anchor_grid_size[0]*2 + 1)
    anchor_y_coords = anchor_y_coords[1:-1:2][:, None].expand(*anchor_grid_size)
    anchor_grid = torch.stack([anchor_x_coords, anchor_y_coords], dim=-1)[None]
    return anchor_grid


@torch.inference_mode()
def run_diffseg_first_iter(anchor_grid: torch.Tensor, 
                           aggregated_self_attention_map: torch.Tensor, 
                           tau: float):
    """
    First iteration of DiffSeg.

    Arguments:
    - anchor_grid: coordinates of each anchor point (1, height_of_anchor_grid, width_of_anchor_grid, 2)
    - aggregated_self_attention_map: (b, h1, w1, h2, w2)
    - tau: distance threshold

    Returns:
    - new_seg_proposals: (b, n_proposals, h2, w2)
    """

    B, H1, W1, H2, W2 = aggregated_self_attention_map.size()

    # grid sample with anchor points to obtain segmentation proposals
    aggregated_self_attention_map = einops.rearrange(
        aggregated_self_attention_map,
        "b h1 w1 h2 w2 -> b (h2 w2) h1 w1"
    )

    anchor_grid = anchor_grid.expand(aggregated_self_attention_map.size(0), -1, -1, -1)
    seg_proposals = F.grid_sample(aggregated_self_attention_map, anchor_grid, mode="nearest", align_corners=True)
    
    aggregated_self_attention_map = einops.rearrange(
        aggregated_self_attention_map,
        "b (h2 w2) h1 w1 -> b (h1 w1) h2 w2",
        h2=H2
    )
    seg_proposals = einops.rearrange(
        seg_proposals,
        "b (h2 w2) ah aw -> b (ah aw) h2 w2",
        h2=H2
    )

    new_seg_proposals = []

    # first iteration of DiffSeg
    for b in range(B):
        new_seg_proposals_ = []

        # compute distance of pairs of proposals and all attention maps.
        # We do not vectorize this operation, since our computation budget is limited.
        for i in range(seg_proposals.size(1)):
            attn_dist = diffseg_compute_attention_distance(seg_proposals[b, i:i+1], aggregated_self_attention_map[b])
            new_seg_proposals_.append(aggregated_self_attention_map[b][attn_dist < tau].mean(dim=0))

        new_seg_proposals_ = torch.stack(new_seg_proposals_, dim=0)
        new_seg_proposals.append(new_seg_proposals_)

    # stack over batch dimension
    new_seg_proposals = torch.stack(new_seg_proposals, dim=0)
    return new_seg_proposals

@torch.inference_mode()
def run_diffseg_post_iter(seg_proposals, tau):
    """
    Arguments:
    - seg_proposals: (b, n_proposals, h2, w2)
    - tau: float (distance threshold)
    """

    B, N, H, W = seg_proposals.size()

    new_seg_proposals = []
    for b in range(B):
        seg_proposals_ = seg_proposals[b]
        new_seg_proposals_ = []

        while seg_proposals_.size(0) > 0:
            sp = seg_proposals_[0]
            attn_dist = diffseg_compute_attention_distance(sp, seg_proposals_)
            new_seg_proposals_.append(seg_proposals_[attn_dist < tau].mean(dim=0))
            seg_proposals_ = seg_proposals_[~(attn_dist < tau)]

        new_seg_proposals_ = torch.stack(new_seg_proposals_)
        new_seg_proposals.append(new_seg_proposals_)

    new_seg_proposals = torch.stack(new_seg_proposals, dim=0)
    return new_seg_proposals


@torch.inference_mode()
def run_diffseg(attention_store: Optional[Dict[str, list]] = None, 
                pipe: Optional[StableDiffusionPipeline] = None, 
                latents: Optional[torch.Tensor] = None, 
                t: Optional[Union[int, torch.Tensor]] = None, 
                prompt_embeds: Optional[torch.Tensor] = None, 
                anchor_grid_size: Iterable[int] = (16, 16), 
                target_size: Iterable[int] = (92, 92), 
                tau: float = 1.0, 
                n_iters: int = 10):
    
    if attention_store is None:
        assert pipe is not None and latents is not None and t is not None and prompt_embeds is not None
        if type(t) != torch.Tensor:
            t = torch.tensor(t).to(pipe.device)

        a_t = pipe.scheduler.alphas_cumprod[t]
        b_t = 1 - a_t
        noise = torch.randn_like(latents)
        latents_t = latents * (a_t ** 0.5) + noise * (b_t ** 0.5)

        attention_store = SimpleAttentionStore()
        set_attention_processors(pipe, attention_store)
        pipe.unet(latents_t, t, prompt_embeds)
        attention_store = attention_store.to_dict()
        
    aggregated_self_attention_map = diffseg_aggregate_self_attention_map(attention_store)
    aggregated_self_attention_map = aggregated_self_attention_map.to(pipe.device)

    anchor_grid = diffseg_initialize_anchor_grid(anchor_grid_size)
    anchor_grid = anchor_grid.to(pipe.device, aggregated_self_attention_map.dtype)
    
    seg_proposals = run_diffseg_first_iter(anchor_grid, aggregated_self_attention_map, tau)
    print(f"Initial num of proposals: {seg_proposals.size(1)}")
    n_proposals = seg_proposals.size(1)

    for i in range(n_iters - 1):
        seg_proposals = run_diffseg_post_iter(seg_proposals, tau)
        print(f"Num of proposals (iter {i + 2}): {seg_proposals.size(1)}")
        if n_proposals == seg_proposals.size(1):
            break

        n_proposals = seg_proposals.size(1)

    seg_proposals = F.interpolate(seg_proposals, size=target_size, mode="bilinear", align_corners=False)
    seg_proposals = seg_proposals.argmax(dim=1)
    return seg_proposals