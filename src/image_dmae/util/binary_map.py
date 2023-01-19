from torch import nn
import torch
from timm.models.layers import to_2tuple

class PatchBinaryMap(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size:int,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
        )
        self.input_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        
        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]

        kernel_size = list(patch_size)
        self.pool = nn.AvgPool2d(kernel_size=kernel_size,
                                 stride=kernel_size)
    
    def forward(self, x):
        B, C, H0, W0 = x.shape # C = 1
        assert (
            H0 == self.img_size[0] and W0 == self.img_size[1]
        ), f"Input image size ({H0}*{W0}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.pool(x).flatten(2) # B, C, H * W
        x = torch.einsum("bcl->blc", x) # [B, H*W, 1]
        assert x.shape[-1] == 1
        return x