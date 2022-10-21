from typing import Iterable
import torch
import mae.util.decoder.transform as transform

def spatial_sampling_v2(
    frames: torch.Tensor,
    spatial_idx: int,
    crop_size: int,
    jitter_scale: Iterable[float]=None,
    jitter_aspect: Iterable[float]=None,
    random_horizontal_flip: bool=None,
):
    """
    Performs spatial sampling from a collection of video frames.
    spatial_idx and crop_size determine the augmentation strategy and resultant frame size.
    jitter and flipping are optional and only used (and only allowed) when spatial_idx=-1 (random cropping)

    If spatial_idx is -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.

    Args:
        frames (torch.Tensor): Torch tensor of input frames
        spatial_idx (int): Determines cropping strategy. (-1, 0, 1, 2)
        crop_size (int): final crop resolution
        jitter_scale (Iterable[float], optional): random_resized_crop scale. Defaults to None.
        jitter_aspect (Iterable[float], optional): random_resized_crop aspect ratio. Defaults to None.
        random_horizontal_flip (bool, optional): enable horizontal flip. Defaults to None.

    Returns:
        torch.Tensor: spatially sampled frames.
    """
    # Ensure spatial_idx are valid options
    assert spatial_idx in [-1, 0, 1, 2], f"spatial_idx({spatial_idx}) not in [-1, 0, 1, 2]."
    # Check random spatial sampling parameters are set when spatial_idx is not -1
    assert ((spatial_idx != -1)
            == ((not jitter_scale) and (not jitter_aspect) and (not random_horizontal_flip)),
            (f"spatial_idx({spatial_idx}) does not match "
             f"jitter_scale({jitter_scale}) jitter_aspect({jitter_aspect}) horizontal_flip({random_horizontal_flip})"))
    if spatial_idx == -1:
        if jitter_scale and jitter_aspect:
            frames = transform.random_resized_crop(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=jitter_scale,
                ratio=jitter_aspect,
            )
        else:
            frames, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # this only resizes it to crop_size
        frames, _ = transform.random_short_side_scale_jitter(frames, crop_size, crop_size)
        # uniform_crop
        frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames