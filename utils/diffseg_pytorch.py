import numpy as np
import torch, einops
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Iterable
from skimage.filters import threshold_otsu


@torch.inference_mode()
def diffseg_aggregate_self_attention_map(self_attention_maps, device=torch.device("cpu"), dtype=torch.float32):
    """
    Aggregate self attention maps by taking self attention maps from multiple layers of diffusion models.

    Arguments:
    - self_attention_maps: list of self attention tensors whose shape are (b, heads, h1, w1, h2, w2).
                           Each tensor represents a self attention maps of a specific layer of diffusion models
    - device: torch device
    - dtype: torch dtype

    Returns:
    - aggregated_self_attention_map: aggregated self attention map (b, h1, w1, h2, w2)
    """

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
    """
    Aggregate cross attention maps extracted from multiple layers of diffusion models.

    Arguments:
    - cross_attention_maps: list of cross attention tensors whose shape are (b, heads, h1, w1, l),
                            where l denotes the number of tokens (l=77)
                            Each element of the list represents a cross attention map of a specific layer in diffusion models
    - attn_token_indices: list of token indices, such as [list_of_token_indices_of_object1, list_of_token_indices_of_object2, ...]
                          For instance, given a prompt "a pomeranian standing on a lush green field",
                          attn_token_indices can be [[2, 3], [8, 9]], where [2, 3] is index list of "pomeranian" (pomeranian is encoded as 2 words by diffusion tokenizer)
                          and [8, 9] denotes index list of "green field". Note that index 0 is the start-of-sentence token.
    - device: torch device
    - dtype: torch dtype

    Returns:
    - aggregated_cross_attention_map: aggregated cross attention map, shape of (b, h1, w1, len(attn_token_indices)), where len(attn_token_indices) represents the number of interested objects.
    - aggregated_cross_attention_mask: cross attention mask computed from aggregated_cross_attention_map by applying otsu thresholding. (b, h1, w1, len(attn_token_indices))
    """

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
def diffseg_compute_attention_distance(attention_map1, attention_map2):
    """
    Compute KL distance between all pairs of attention map1 and attention map2. (eq. 5 in the paper)

    Arguments:
    - attention_map1: (m h w). m spatial attention maps. Each attention map should have shape (h, w).
    - attention_map2: (n h w). n spatial attention maps. Each attention map should have shape (h, w).

    Returns:
    - D: (m, n). pairwise KL distance between all pairs of attention maps from attention_map1 and attention_map2.
    """

    log_p = attention_map1.log().flatten(1)
    log_q = attention_map2.log().flatten(1)

    p = attention_map1.flatten(1)
    q = attention_map2.flatten(1)

    forward_kl = torch.sum(p * log_p, dim=-1, keepdim=True) - p @ log_q.t()
    reverse_kl = torch.sum(q * log_q, dim=-1, keepdim=True).t() - (q @ log_p.t()).t()
    D = (forward_kl + reverse_kl)/2
    return D


@torch.inference_mode()
def diffseg_initialize_anchor_grid(anchor_grid_size: Iterable[int]):
    """
    get initial anchor points for obtaining initial proposals

    Arguments:
    - anchor_grid_size: tuple of int (height, width)

    Returns:
    - anchor_grid: coordinates of each anchor point (1, height_of_anchor_grid, width_of_anchor_grid, 2)
    """

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
    """
    First iteration of DiffSeg.

    Refine the proposals by aggregating similar self attention maps

    Arguments:    
    - anchor_grid: (1, height_of_anchor_grid, width_of_anchor_grid, 2). grid for sampling initial proposals
    - aggregated_self_attention_map: (b, h1, w1, h2, w2). aggregated self attention map by diffseg_aggregate_self_attention_map.
    - tau: KL distance threshold for determining if two attention maps are similar.
    - batch_size: batch_size for computing attention distance. In the first iteration, 
                  since the computational cost of simultaneously computing KL distance between all pairs of ths proposals and the attention maps is very high,
                  we divide the computation into batches.

    Returns:
    - new_seg_proposals: list of refined proposals. Each proposal has shape of (num_of_proposals, h2, w2).
                         Length of this list is the same as the number of sample (=len(aggregated_self_attention_map)).
    """
    
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
    Rest iteration of DiffSeg.
    Refine the proposals by merging similar proposals

    Arguments:
    - seg_proposals: list of proposals of shape (n_proposals, h2, w2).
    - tau: float (distance threshold)

    Returns:
    - new_seg_proposals: list of refined proposals of shape (n_proposals, h2, w2).
                         n_proposals is less than or equal to the number of proposals in seg_proposals
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
    run DiffSeg.

    Arguments:
    - self_attention_maps: list of self attention map from multiple layers of diffusion models.
    - cross_attention_maps: list of cross attention map from multiple layers of diffusion models.
    - anchor_grid_size: tuple of height and width of anchor grid.
    - target_size: desirable height and width of final segmentation map.
    - tau: float (distance threshold)
    - n_iters: the number of iterations of DiffSeg iterations. 1 + the_number_of_iterations_of_post_iter
    - batch_size: The batch size used in computing KL distance. THIS IS NOT the batch size of inputs of diffusion models.
    - device: torch device
    - dtype: torch dtype
    - attn_token_indices: list of token indices, such as [list_of_token_indices_of_object1, list_of_token_indices_of_object2, ...]
                          For instance, given a prompt "a pomeranian standing on a lush green field",
                          attn_token_indices can be [[2, 3], [8, 9]], where [2, 3] is index list of "pomeranian" (pomeranian is encoded as 2 words by diffusion tokenizer)
                          and [8, 9] denotes index list of "green field". Note that index 0 is the start-of-sentence token.
                          If add_semantic is set True, this argument is required.
    - add_semantic: bool. whether to add semantic information to the segmentation map.
                    If this is set to True, attn_token_indices is required.
    """

    aggregated_self_attention_map = diffseg_aggregate_self_attention_map(self_attention_maps, device, dtype)

    anchor_grid = diffseg_initialize_anchor_grid(anchor_grid_size)
    anchor_grid = anchor_grid.to(device, dtype)
    
    seg_proposals = run_diffseg_first_iter(anchor_grid, aggregated_self_attention_map, tau, batch_size)
    for i in range(n_iters - 1):
        seg_proposals = run_diffseg_post_iter(seg_proposals, tau)

    # add_semantic: this is not the implementation of the DiffSeg paper.
    # This code will be fixed later.
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