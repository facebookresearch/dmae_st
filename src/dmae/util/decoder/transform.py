# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
import random

import numpy as np
import torch

def resized_crop(
    images,
    target_height,
    target_width,
    params
):
    """
    Crop the given images to random size and aspect ratio. A crop of random
    size (default: of 0.08 to 1.0) of the original size and a random aspect
    ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This
    crop is finally resized to given size. This is popularly used to train the
    Inception networks.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        params: Parameters to apply the crop
    """
    i, j, h, w = params
    cropped = images[:, :, i : i + h, j : j + w]
    images =  torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return images

def random_resized_crop(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
    return_params=False
):
    """
    Crop the given images to random size and aspect ratio. A crop of random
    size (default: of 0.08 to 1.0) of the original size and a random aspect
    ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This
    crop is finally resized to given size. This is popularly used to train the
    Inception networks.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """

    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, :, i : i + h, j : j + w]
    images =  torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    if return_params:
        return images, (i, j, h, w)
    return images

def horizontal_flip(images):
    """
    Perform horizontal flip on the given images.
    Args:
        images (tensor): images to perform horizontal flip, the dimension is
            ... x `height` x `width`.
    Returns:
        images (tensor): images with dimension of
            ... x `height` x `width`.
    """
    images = images.flip((-1))
    return images

def random_horizontal_flip(prob, images, return_params=False):
    """
    Perform random horizontal flip on the given images.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            ... x `height` x `width`.
    Returns:
        images (tensor): images with dimension of
            ... x `height` x `width`.
    """

    flip = False
    if np.random.uniform() < prob:
        flip = True
        images = images.flip((-1))

    if return_params:
        return images, flip
    return images
    
def _get_param_spatial_crop(
    scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False
):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w
