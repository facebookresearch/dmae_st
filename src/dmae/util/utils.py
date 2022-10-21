import math
import torch


def compute_masking_ratios(
    mask: torch.Tensor,
    patched_binary_map: torch.Tensor,
    num_patches: int
):
    N, L = mask.shape
    
    patched_binary_map = patched_binary_map.view(N, L)
    patches_in, patches_out = patched_binary_map >= 0.5, patched_binary_map < 0.5
    n_empty = (patched_binary_map == 0.).all(dim=-1).type(torch.float32).sum()
    
    mask = mask.type(torch.bool)
    
    n_patches_in = patches_in.type(torch.float32).sum(-1)
    n_patches_out = patches_out.type(torch.float32).sum(-1)
    
    n_mask_in = torch.logical_and(mask, patches_in).type(torch.float32).sum(-1)
    n_mask_out = torch.logical_and(mask, patches_out).type(torch.float32).sum(-1)
    
    return (
        (n_patches_in / num_patches).mean(),
        (n_mask_in / n_patches_in).nanmean(),
        (n_patches_out / num_patches).mean(),
        (n_mask_out / n_patches_out).nanmean(),
        n_empty
    )
    
def get_temperature(args, cur_epoch=None):
    if args.temperature_schedule == "const":
        return args.temperature
    elif args.temperature_schedule == "cos":
        mask_ratio_start = args.temperature
        mask_ratio_end = args.temperature_end
        num_epoch = args.epochs
        return (
            mask_ratio_end
            + (mask_ratio_start - mask_ratio_end)
            * (math.cos(math.pi * cur_epoch / num_epoch) + 1.0)
            * 0.5
        )
