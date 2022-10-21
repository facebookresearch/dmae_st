
import argparse
import os
import torch
from dmae.models_dmae import dmae_vit_large_patch16
from dmae.util.kinetics_bbox import KineticsBbox
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid
from torchvision.io import write_png


parser = argparse.ArgumentParser("Output-DMAKS")
parser.add_argument("--temperature", default=1.0, type=float)

if __name__ == "__main__":
    args = parser.parse_args()
    
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
    
    model = dmae_vit_large_patch16(
        img_size=224,
        in_chans=3,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        t_patch_size=1,
        mask_type="directed",
        learnable_pos_embed=True,
        sep_pos_embed=True,
        cls_embed=True,
        pred_t_dim=16,
        temperature=args.temperature,
    ).cuda()
    
    for idx in range(1000, 30000, 1000):
        print(idx)
        sample, bbox, _ = ds.__getitem__(idx) # 2, C, T, H, W
        
        masks = []
        ratios = [0.3, 0.6, 0.75, 0.9]
        for mask_ratio_idx, mask_ratio in enumerate(ratios):
            print(idx, mask_ratio_idx)
            with torch.no_grad():
                _, _, _, comp = model(sample.cuda(), bbox.cuda(), mask_ratio, visualize=True)
                masked = comp[:, 1].permute(0, 2, 1, 3, 4).cpu()
                masks.append(masked)
        
        bbox = bbox.tile(1, 3, 1, 1, 1) # 2, C, T, H, W
        sample = sample.permute(0, 2, 1, 3, 4) # 2, T, C, H, W
        bbox = bbox.permute(0, 2, 1, 3, 4)
        third = sample * bbox
        
        masks = [x[:, :, None] for x in masks]
        
        combined = torch.cat((sample[:, :, None],
                              bbox[:, :, None],
                              third[:, :, None],
                              *masks),
                             dim=2).flatten(0, 2)
        
        combined = resize(combined, 200) * 255.
        
        N = combined.shape[0]
        img_grid = make_grid(combined.type(torch.uint8), nrow=len(ratios) + 3)
        
        f = f"experiments/dmae-st/SAMPLE/dmask/img/{idx}.png"
        dir = os.path.dirname(f)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        write_png(img_grid, f)