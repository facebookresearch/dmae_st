# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import mae.util.logging as logging
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mae.util import video_vit
from mae.util.logging import master_print as print
from mae.util.pos_embed import get_3d_sincos_pos_embed
from timm.models.layers import to_2tuple

class PatchBinaryMap(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size:int,
        frames: int,
        t_patch_size: int
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        
        self.img_size = img_size
        self.patch_size = patch_size

        self.frames = frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        kernel_size = [t_patch_size] + list(patch_size)
        self.pool = nn.AvgPool3d(kernel_size=kernel_size,
                                 stride=kernel_size)
    
    def forward(self, x):
        B, C, T, H, W = x.shape # C = 1
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.frames
        
        x = self.pool(x).flatten(3)
        x = torch.einsum("bcts->btsc", x) # [B, T, H*W, 1]
        assert x.shape[-1] == 1
        return x
        
        
class DirectedMaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=4,
        patch_embed=video_vit.PatchEmbed,
        patch_binary_map=PatchBinaryMap,
        # encoder_block=video_vit.BlockAttn,
        # decoder_block=video_vit.BlockAttn,
        encoder_attn="AttentionWithCls",
        decoder_attn="AttentionOrg",
        mask_type=False,
        no_qkv_bias=False,
        learnable_pos_embed=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        pred_t_dim=8,
        # temperature=1.0,
        flip_mask=False,
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        # encoder_attn_func = video_vit.__dict__[encoder_attn]
        # decoder_attn_func = video_vit.__dict__[decoder_attn]
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames
        # self.temperature = temperature
        self.flip_mask=flip_mask

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        self.patch_binary_map = patch_binary_map(
            img_size=img_size,
            patch_size=patch_size,
            frames=num_frames,
            t_patch_size=t_patch_size
        )
        
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size
        self.mask_type = mask_type
        self.learnable_pos_embed = learnable_pos_embed

        print(f"num_patches {num_patches}")

        decoder_attn_func = video_vit.__dict__[decoder_attn]
        encoder_attn_func = video_vit.__dict__[encoder_attn]

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            if learnable_pos_embed:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, _num_patches, embed_dim), requires_grad=True
                )  # fixed sin-cos embedding
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, _num_patches, embed_dim), requires_grad=False
                )  # fixed sin-cos embedding

        assert "RelPos" not in encoder_attn, "Not support RelPos for MAE model"
        encoder_block = video_vit.BlockAttn
        attn_func = video_vit.__dict__[encoder_attn]
        self.blocks = nn.ModuleList(
            [
                encoder_block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    attn_func=encoder_attn_func,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            if learnable_pos_embed:
                self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, _num_patches, decoder_embed_dim), requires_grad=True
                )  # fixed sin-cos embedding
            else:
                self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, _num_patches, decoder_embed_dim), requires_grad=False
                )  # fixed sin-cos embedding

        decoder_block = video_vit.BlockAttn
        self.decoder_blocks = nn.ModuleList(
            [
                decoder_block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    attn_func=decoder_attn_func,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size ** 2 * in_chans,
            bias=True,
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            if self.learnable_pos_embed:
                # torch.nn.init
                torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
            else:
                pos_embed = get_3d_sincos_pos_embed(
                    self.pos_embed.shape[-1],
                    self.patch_embed.grid_size,
                    self.patch_embed.t_grid_size,
                    cls_token=self.cls_embed,
                )
                self.pos_embed.data.copy_(
                    torch.from_numpy(pos_embed).float().unsqueeze(0)
                )

                decoder_pos_embed = get_3d_sincos_pos_embed(
                    self.decoder_pos_embed.shape[-1],
                    self.patch_embed.grid_size,
                    self.patch_embed.t_grid_size,
                    cls_token=self.cls_embed,
                )
                self.decoder_pos_embed.data.copy_(
                    torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
                )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            # torch.nn.init.normal_(self.cls_token, std=.02)
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, C, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, C, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p ** 2 * C))
        self.patch_info = (N, T, H, W, p, u, t, h, w, C)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w, C = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 3))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 3, T, H, W))
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def weighted_masking(self, x, weight, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device) * weight[..., 0]  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, binary_map, mask_ratio, temperature):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        if self.mask_type == "directed":
            assert binary_map is not None
            
            if self.flip_mask:
                binary_map = 1. - binary_map
            
            binary_map = binary_map + 1.0 # Add 1 so that low-pri regions aren't 0. Rely on temperature to modulate
            patch_binary_map = self.patch_binary_map(binary_map) # [B, T, H*W, 1]
            patch_weights = torch.softmax(patch_binary_map[..., 0] / temperature,
                                          dim=-1)[..., None].reshape(N, T * L, 1)

        if self.mask_type in ["st", "directed"]:
            x = x.reshape(N, T * L, C)
        elif self.mask_type == "t":
            x = x.reshape(N * T, L, C)
        elif self.mask_type == "tube":
            x = torch.einsum("ntlc->nltc", x.reshape([N, T, L, C])).reshape(
                [N, L, T * C]
            )
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")

        # masking: length -> length * mask_ratio
        if self.mask_type == "directed":
            x, mask, ids_restore, ids_keep = self.weighted_masking(x, patch_weights, mask_ratio)
        else:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        if self.mask_type in ["st", "t", "directed"]:
            x = x.view(N, -1, C)
        elif self.mask_type == "tube":
            _, L_new, _ = x.shape
            x = x.reshape([N, L_new, T, C])
            x = torch.einsum("nltc->ntlc", x)
            x = x.reshape([N, T * L_new, C])
            # N 1 L C -> N T L C
            mask = mask.repeat(1, T, 1, 1)
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        if self.mask_type == "st" or self.mask_type == "directed":
            x_ = x_.view([N, T * H * W, C])
        elif self.mask_type == "t":
            x_ = x_.view([N * T, H * W, C])
        elif self.mask_type == "tube":
            x_ = x_.reshape([N, T, H * W, C])
            x_ = torch.einsum("ntlc->nltc", x_)
            x_ = x_.reshape([N, H * W, T * C])
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        if self.mask_type in ["st", "t", "directed"]:
            x = x_.view([N, T * H * W, C])
        elif self.mask_type == "tube":
            x = x_.reshape([N, H * W, T, C])
            x = torch.einsum("nltc->ntlc", x)
            x = x.reshape([N, T * H * W, C])
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")

        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_loss(self, imgs, pred, mask, visualize):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.pred_t_dim,
            )
            .long()
            .to(imgs.device),
        )
        target = self.patchify(_imgs)
        if visualize:
            self.target = target
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, binary_map=None, mask_ratio=0.75, temperature=1.0, visualize=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, binary_map, mask_ratio, temperature)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, visualize)

        if visualize:
            N, T, H, W, p, u, t, h, w, C = self.patch_info
            reconstruct = self.unpatchify(pred)
            masked = self.unpatchify(self.target * (1 - mask.reshape(N, t * h * w, 1)))
            
            comparison = torch.stack(
                [imgs, masked, reconstruct],
                dim=1,
            )
            return loss, pred, mask, comparison
        else:
            return loss, pred, mask, torch.Tensor()
        

def dmae_vit_base_patch16(**kwargs):
    model = DirectedMaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def dmae_vit_large_patch16(**kwargs):
    model = DirectedMaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def dmae_vit_huge_patch14(**kwargs):
    model = DirectedMaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_fn  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_fn  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14  # decoder: 512 dim, 8 blocks

class DirectedMaskedAutoencoderViTV2(DirectedMaskedAutoencoderViT):
    def forward_encoder(self, x, binary_map, mask_ratio, temperature):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        if self.mask_type == "directed":
            assert binary_map is not None
            
            if self.flip_mask:
                binary_map = 1. - binary_map
            
            # binary_map = binary_map + 1.0 # Add 1 so that low-pri regions aren't 0. Rely on temperature to modulate
            patch_binary_map = self.patch_binary_map(binary_map) # [B, T, H*W, 1]
            patch_weights = torch.softmax(patch_binary_map[..., 0] * temperature,
                                          dim=-1)[..., None].reshape(N, T * L, 1)

        if self.mask_type in ["st", "directed"]:
            x = x.reshape(N, T * L, C)
        elif self.mask_type == "t":
            x = x.reshape(N * T, L, C)
        elif self.mask_type == "tube":
            x = torch.einsum("ntlc->nltc", x.reshape([N, T, L, C])).reshape(
                [N, L, T * C]
            )
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")

        # masking: length -> length * mask_ratio
        if self.mask_type == "directed":
            x, mask, ids_restore, ids_keep = self.weighted_masking(x, patch_weights, mask_ratio)
        else:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        if self.mask_type in ["st", "t", "directed"]:
            x = x.view(N, -1, C)
        elif self.mask_type == "tube":
            _, L_new, _ = x.shape
            x = x.reshape([N, L_new, T, C])
            x = torch.einsum("nltc->ntlc", x)
            x = x.reshape([N, T * L_new, C])
            # N 1 L C -> N T L C
            mask = mask.repeat(1, T, 1, 1)
        else:
            raise NotImplementedError(f"not supported mask type {self.mask_type}")

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore
        
    
def dmae_vit_base_patch16_v2(**kwargs):
    model = DirectedMaskedAutoencoderViTV2(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def dmae_vit_large_patch16_v2(**kwargs):
    model = DirectedMaskedAutoencoderViTV2(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def dmae_vit_huge_patch14_v2(**kwargs):
    model = DirectedMaskedAutoencoderViTV2(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model