# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import torch
from dmae.util.kinetics_bbox import KineticsBbox
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid
from torchvision.io import write_png


if __name__ == "__main__":
    ds = KineticsBbox(
        mode="pretrain",
        path_to_data_dir="/datasets01/kinetics_400/k400",
        path_to_bbox_dir="data/kinetics_400_annotations-videos",
        num_retries=12,
        sampling_rate=4,
        num_frames=16,
        crop_size=224,
        repeat_aug=2,
        pre_crop_size=256,
        random_horizontal_flip=True,
        jitter_aspect_relative=[3./4., 4./3.],
        jitter_scale_relative=[0.5, 1.0],
        backend="torchvision",
        norm_mean=(0., 0., 0.),
        norm_std=(1., 1., 1.,)
    )
    
    for idx in range(0, 5000, 100):
        sample, bbox, _ = ds.__getitem__(idx)
        bbox = bbox.tile(1, 3, 1, 1, 1) # 2, C, T, H, W
        
        sample = sample.permute(0, 2, 1, 3, 4) # 2, T, C, H, W
        bbox = bbox.permute(0, 2, 1, 3, 4)
        third = sample * bbox
        
        combined = torch.cat((sample[:, :, None],
                              bbox[:, :, None],
                              third[:, :, None]),
                             dim=2).flatten(0, 2)
        
        combined = resize(combined, 200) * 255.
        
        N = combined.shape[0]
        img_grid = make_grid(combined.type(torch.uint8), nrow=3)
        
        f = f"experiments/dmae-st/SAMPLE/bbox/img/{idx}.png"
        dir = os.path.dirname(f)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        write_png(img_grid, f)