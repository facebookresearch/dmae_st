#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pickle as pkl
from typing import Any, Callable, Dict, Optional

from torchvision.transforms import functional as tvF
import torch
from dmae.util.decoder.bboxes import bbox_to_binary_map

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class ImagenetBbox(ImageFolder):
    def __init__(
        self,
        root: str,
        bbox_root: str,
        spatial_transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root, None, target_transform, loader, is_valid_file
        )
        
        self.spatial_transform = spatial_transform
        self.image_transform = image_transform
        
        self.bbox_root = bbox_root
        self.idx_to_class = {cls:idx for (idx, cls) in self.class_to_idx.items()}
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        class_name = self.idx_to_class[target]
        file_name = path.split("/")[-1]
        
        # load image 
        sample = self.loader(path) # C, H0, W0
        sample = tvF.to_tensor(sample)
        C, H0, W0 = sample.shape
        
        bbox_path = os.path.join(self.bbox_root, f"{class_name}.pkl")
        with open(bbox_path, 'rb') as bbox_f:
            bbox_dict = pkl.load(bbox_f)
            bboxes = bbox_dict[file_name][:, :4]
            
        bbox_binary_map = bbox_to_binary_map(bboxes, sample.shape[-2:]) # H, W0, 1
        bbox_binary_map = bbox_binary_map.permute(2, 0, 1) # 1, H, W0
        
        if self.spatial_transform is not None:
            stack = torch.cat((sample, bbox_binary_map)) # C + 1, H, W0
            stack = self.spatial_transform(stack)
            sample, bbox_binary_map = stack[:C], stack[C:]
        
        if self.image_transform is not None:
            sample = self.image_transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, bbox_binary_map, target
        