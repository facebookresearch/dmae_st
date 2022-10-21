# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
from functools import partial
from re import A

import torch
from torch import nn

from model.base.transformers import Block, get_pos_embed, init_tsfmr_weights


class PoseClassificationTransformer(nn.Module):
    """ Pose Classification Transformer
    
    Takes sequence input and outputs a class prediction.
    """

    def __init__(self, num_frames: int, num_skeletons: int, num_joints: int, input_dim: int,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None,
                 hybrid_pos_embed=False):
        """
        Args:
            num_frames (int): number of frames per example
            num_skeletons (int): number of skeletons per example
            num_joints (int): number of joints per example
            input_dim (int): input token dimension
            seq_len (int): sequence length 
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module) activation layer
        """

        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.ReLU
        self.embed_layer = nn.Linear(input_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # get_pos_embed returns (1, embed_len, embed_dim)
        if hybrid_pos_embed:
            learnable_embed = torch.zeros(1, num_frames * num_skeletons * num_joints + 1, 3 * (embed_dim // 4)) # (1, N+1 3D/4)
            pos_total_a, pos_embed_frames = get_pos_embed(num_frames + 1, embed_dim // 4).split((1, num_frames), dim=1)
                
            pos_embed_frames = pos_embed_frames[:, :, None, None, :].tile(1, 1, num_skeletons, num_joints, 1).flatten(1, -2) # (1, N, D // 4)
            pos_embed_frames = torch.cat((pos_total_a, pos_embed_frames), dim=1)# (1, N+1, D//4)
            
            self.pos_embed = nn.Parameter(torch.cat((learnable_embed, pos_embed_frames), dim=-1), requires_grad=False) # (1, N=1, D)
            self.pos_embed[:, :, :3 * (embed_dim // 4)].requires_grad = True
        else:
            pos_total_a, pos_embed_frames = get_pos_embed(num_frames + 1, embed_dim // 2).split((1, num_frames), dim=1)
            pos_total_b, pos_embed_skeletons = get_pos_embed(num_skeletons + 1, embed_dim // 4).split((1, num_skeletons), dim=1)
            pos_total_c, pos_embed_joints = get_pos_embed(num_joints + 1, embed_dim // 4).split((1, num_joints), dim=1)
            pos_total = torch.cat((pos_total_a, pos_total_b, pos_total_c), dim=-1) # (1, 1, D)
            
            pos_embed_frames = pos_embed_frames[:, :, None, None, :].tile(1, 1, num_skeletons, num_joints, 1).flatten(1, -2) # (1, N, D // 2)
            pos_embed_skeletons = pos_embed_skeletons[:, None, :, None, :].tile(1, num_frames, 1, num_joints, 1).flatten(1, -2)
            pos_embed_joints = pos_embed_joints[:, None, None, :, :].tile(1, num_frames, num_skeletons, 1, 1).flatten(1, -2)
            pos_embed = torch.cat((pos_embed_frames, pos_embed_skeletons, pos_embed_joints), dim=-1) # (1, N, D)
            pos_embed = torch.cat((pos_total, pos_embed), dim=1) # (1, N+1, D)
            self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None

        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.apply(init_tsfmr_weights)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B, frames, skeletons, joints, C = x.shape
        N = frames * skeletons * joints

        padding_mask = (x == 0.).all(dim=-1).flatten(1, -1).type(torch.float32) # B, N
        padding_mask = torch.cat((torch.zeros((B, 1)).to(x.device), padding_mask), dim=1)
        
        x = self.embed_layer(x).flatten(1, -2) # B, N, D
 
        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.norm(x)

        return self.head(x[:, 0])