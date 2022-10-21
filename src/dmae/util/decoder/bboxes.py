# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from typing import Iterable, Union, Tuple
import numpy as np
import torch


def bbox_to_binary_map(
    bboxes: torch.Tensor,
    map_dim: Iterable[int]
):
    """
    Converts a list of bounding boxes into a binary map of map_dim dimensions.
    Elements in the map are set to 1 if within a bounding box and 0 if without.

    Args:
        bboxes (torch.Tensor): [n, 4] tensor of bounding boxes. [n, xmin, ymin, xmax, ymax]
        map_dim (Iterable[int]]): Iterable of length 2 determing map dimensions.

    Returns:
        torch.Tensor: binary mapping matching map_dim dimension.
    """
    assert len(map_dim) == 2, f"map_dim({map_dim}) not of length 2."
    bin_map = torch.zeros(map_dim[0], map_dim[1])
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox.type(torch.int64)
        bin_map[ymin:ymax+1, xmin:xmax+1] = 1. # add 1 to make the max bound inclusive.
    
    return bin_map[..., None]

if __name__ == "__main__":
    from pprint import pprint
    bboxes = torch.tensor([[0., 1., 2., 3.],
                           [1., 2., 3., 5.],
                           [5., 5., 5., 5.]])
    x = bbox_to_binary_map(bboxes, (6, 6))
    
    ans = torch.tensor([[0., 0., 0., 0., 0., 0.],
                        [1., 1., 1., 0., 0., 0.],
                        [1., 1., 1., 1., 0., 0.],
                        [1., 1., 1., 1., 0., 0.],
                        [0., 1., 1., 1., 0., 0.],
                        [0., 1., 1., 1., 0., 1.],])
    
    pprint(x[..., 0])
    pprint(ans)
    assert torch.equal(x, ans[..., None])
