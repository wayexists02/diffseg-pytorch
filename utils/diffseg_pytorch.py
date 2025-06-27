import numpy as np
import torch, einops
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Iterable
from skimage.filters import threshold_otsu


@torch.inference_mode()
def diffseg_aggregate_self_attention_map(self_attention_maps, device=torch.device("cpu"), dtype=torch.float32):
    self_attention_maps = [am.mean(dim=1).to(device, dtype) for am in self_attention_maps]
    self_attention_map_sizes = list(map(lambda map: map.size()[-1], self_attention_maps))
    self_attention_map_sizes = np.unique(self_attention_map_sizes)

    self_attention_maps_in_sizes = {
        size: []
        for size in self_attention_map_sizes
    }

    for sam in self_attention_maps:
        self_attention_maps_in_sizes[sam.size(-1)].append(sam)

    max_size = max(self_attention_map_sizes)
    aggregated_self_attention_map = 0
    R_sum = 0

    for size, sams in self_attention_maps_in_sizes.items():
        M = len(sams)
        sams = torch.cat(sams, dim=0)
        sams = einops.rearrange(sams, "bm h1 w1 h2 w2 -> bm (h1 w1) h2 w2")
        sams = F.interpolate(sams, size=(max_size, max_size), mode="bilinear", align_corners=False)

        sams = einops.rearrange(sams, "bm (h1 w1) h2 w2 -> bm (h2 w2) h1 w1", h1=size)
        sams = F.interpolate(sams, size=(max_size, max_size), mode="nearest")
        sams = einops.rearrange(sams, "(b m) (h2 w2) h1 w1 -> b m h1 w1 h2 w2", m=M, h2=max_size)

        aggregated_self_attention_map += torch.sum(sams * size, dim=1)
        R_sum += size * M

    aggregated_self_attention_map /= R_sum
    aggregated_self_attention_map = aggregated_self_attention_map / aggregated_self_attention_map.sum(dim=(-1, -2), keepdim=True)
    return aggregated_self_attention_map


@torch.inference_mode()
def diffseg_aggregate_cross_attention_map(cross_attention_maps, attn_token_indices, device=torch.device("cpu"), dtype=torch.float32):
    indices = []
    for i in attn_token_indices:
        indices.extend(i)

    cross_attention_maps = [cam.mean(dim=1)[..., indices].to(device, dtype) for cam in cross_attention_maps]
    sizes = set(map(lambda cam: cam.size(-2), cross_attention_maps))
    max_size = max(sizes)

    cross_attention_maps_in_sizes = {
        size: []
        for size in sizes
    }

    for cam in cross_attention_maps:
        cross_attention_maps_in_sizes[cam.size(-2)].append(cam)

    aggregated_cross_attention_map = 0
    R_sum = 0

    for size, cams in cross_attention_maps_in_sizes.items():
        M = len(cams)
        cams = torch.cat(cams, dim=0)
        cams = einops.rearrange(cams, "bm h w l -> bm l h w")
        cams = F.interpolate(cams, size=(max_size, max_size), mode="nearest")
        cams = einops.rearrange(cams, "(b m) l h w -> b m h w l", m=M)

        aggregated_cross_attention_map += cams.sum(dim=1)
        R_sum += M

    aggregated_cross_attention_map /= R_sum
    aggregated_cross_attention_map = aggregated_cross_attention_map.split([len(ind) for ind in attn_token_indices], dim=-1)
    aggregated_cross_attention_map = torch.stack([acam.mean(dim=-1) for acam in aggregated_cross_attention_map], dim=-1)

    aggregated_cross_attention_map = einops.rearrange(aggregated_cross_attention_map, "b h w l -> (b l) h w")
    aggregated_cross_attention_map_np = aggregated_cross_attention_map.float().cpu().numpy()
    ths = list(map(lambda i: threshold_otsu(aggregated_cross_attention_map_np[i]), range(len(aggregated_cross_attention_map))))
    ths = torch.tensor(ths, device=aggregated_cross_attention_map.device)[:, None, None]

    aggregated_cross_attention_mask = (aggregated_cross_attention_map >= ths).type(aggregated_cross_attention_map.dtype)
    aggregated_cross_attention_mask = einops.rearrange(aggregated_cross_attention_mask, "(b l) h w -> b h w l", l=len(attn_token_indices))
    aggregated_cross_attention_map = einops.rearrange(aggregated_cross_attention_map, "(b l) h w -> b h w l", l=len(attn_token_indices))
    return aggregated_cross_attention_map, aggregated_cross_attention_mask


@torch.inference_mode()
def diffseg_compute_attention_distance(attention_map1: torch.Tensor, attention_map2: torch.Tensor):
    """
    Arguments:
    - attention_map1: (n h w)
    - attention_map2: (m h w)

    Returns:
    - D: (n, m) pairwise KL distance between all pairs of attention maps from attention_map1 and attention_map2
    """

    log_attention_map1 = attention_map1.log()
    log_attention_map2 = attention_map2.log()

    forward_kl = torch.sum(attention_map1[:, None] * (log_attention_map1[:, None] - log_attention_map2[None]), dim=(-1, -2))
    reverse_kl = torch.sum(attention_map2[None] * (log_attention_map2[None] - log_attention_map1[:, None]), dim=(-1, -2))
    D = (forward_kl + reverse_kl)/2
    return D


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
                           tau: float,
                           batch_size: int = 32):
    
    N, H1, W1, H2, W2 = aggregated_self_attention_map.size()
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

    for i in range(N):
        new_seg_proposals_ = []
        num_batches = int(np.ceil(seg_proposals.size(1) / batch_size))
        for b in range(num_batches):
            start_index = b * batch_size
            end_index = min((b + 1)*batch_size, seg_proposals.size(1))

            attn_dist = diffseg_compute_attention_distance(
                seg_proposals[i, start_index:end_index],
                aggregated_self_attention_map[i]
            )

            attn_dist_mask = attn_dist < tau

            index = torch.arange(aggregated_self_attention_map.size(1)).to(aggregated_self_attention_map.device)
            index = index.repeat(end_index - start_index)

            index = index[attn_dist_mask.flatten()]
            new_seg_proposals_batch = aggregated_self_attention_map[i, index].split(attn_dist_mask.type(torch.long).sum(dim=-1).tolist())
            new_seg_proposals_.append(torch.stack([s.mean(dim=0) for s in new_seg_proposals_batch]))
            
        new_seg_proposals_ = torch.cat(new_seg_proposals_, dim=0)
        new_seg_proposals.append(new_seg_proposals_)

    return new_seg_proposals

@torch.inference_mode()
def run_diffseg_post_iter(seg_proposals, tau):
    """
    Arguments:
    - seg_proposals: (b, n_proposals, h2, w2)
    - tau: float (distance threshold)
    """

    N = len(seg_proposals)

    new_seg_proposals = []
    for i in range(N):
        seg_proposals_ = seg_proposals[i]
        new_seg_proposals_ = []

        while seg_proposals_.size(0) > 0:
            sp = seg_proposals_[0]
            attn_dist = diffseg_compute_attention_distance(sp[None], seg_proposals_)[0]
            new_seg_proposals_.append(seg_proposals_[attn_dist < tau].mean(dim=0))
            seg_proposals_ = seg_proposals_[~(attn_dist < tau)]

        new_seg_proposals_ = torch.stack(new_seg_proposals_)
        new_seg_proposals.append(new_seg_proposals_)

    return new_seg_proposals


@torch.inference_mode()
def run_diffseg(self_attention_maps: List[torch.Tensor],
                cross_attention_maps: List[torch.Tensor],
                anchor_grid_size: Iterable[int] = (16, 16), 
                target_size: Iterable[int] = (92, 92), 
                tau: float = 1.0, 
                n_iters: int = 10,
                batch_size: int = 32,
                device = torch.device("cpu"),
                dtype = torch.float16,
                attn_token_indices: Optional[List[int]] = None,
                add_semantic: bool = False):
    """
    Arguments:
    - self_attention_maps: list of self attention map with various spatial size
    """

    aggregated_self_attention_map = diffseg_aggregate_self_attention_map(self_attention_maps, device, dtype)

    anchor_grid = diffseg_initialize_anchor_grid(anchor_grid_size)
    anchor_grid = anchor_grid.to(device, dtype)
    
    seg_proposals = run_diffseg_first_iter(anchor_grid, aggregated_self_attention_map, tau, batch_size)
    for i in range(n_iters - 1):
        seg_proposals = run_diffseg_post_iter(seg_proposals, tau)

    if add_semantic is True:
        _, aggregated_cross_attention_mask = diffseg_aggregate_cross_attention_map(cross_attention_maps, attn_token_indices, device, dtype)
        B, H, W, L = aggregated_cross_attention_mask.size()

        bg_mask = (aggregated_cross_attention_mask.sum(dim=-1) < 0.5).type(aggregated_cross_attention_mask.dtype)
        aggregated_cross_attention_mask_oh = torch.cat([
            bg_mask.unsqueeze(-1), aggregated_cross_attention_mask
        ], dim=-1)

        seg_proposals_with_segmentic = []

        for i in range(len(seg_proposals)):
            seg_proposals_i = seg_proposals[i] # (num_proposals, h, w)
            semantic = seg_proposals_i.flatten(-2, -1).mm(aggregated_cross_attention_mask_oh[i].flatten(0, 1)).argmax(dim=-1) # majority voting (num_seg_proposals,)
            seg_proposals_with_segmentic_i = [seg_proposals_i[semantic == 0]] + [seg_proposals_i[semantic == s].sum(dim=0, keepdim=True) for s in range(1, len(attn_token_indices) + 1)]
            seg_proposals_with_segmentic_i = torch.cat(seg_proposals_with_segmentic_i)
            seg_proposals_with_segmentic.append(seg_proposals_with_segmentic_i)

        seg_proposals = seg_proposals_with_segmentic

    seg_proposals = [F.interpolate(seg_proposals[i][None], size=target_size, mode="bilinear", align_corners=False)[0] for i in range(len(seg_proposals))]
    seg_proposals = torch.stack([seg_proposals[i].argmax(dim=0) for i in range(len(seg_proposals))])
    return seg_proposals