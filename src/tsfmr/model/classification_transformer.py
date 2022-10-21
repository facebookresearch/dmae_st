
import math
from functools import partial
from re import A

import torch
from torch import nn

from model.base.transformers import Block, init_tsfmr_weights


class ClassificationTransformer(nn.Module):
    """ Classification Transformer
    
    Takes sequence input and outputs a class prediction.
    """

    def __init__(self, input_dim: int, seq_len: int, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None):
        """
        Args:
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
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

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
        B, N, C = x.shape
        padding_mask = (x == 0.).all(dim=-1).type(torch.float32) # B, N
        padding_mask = torch.cat((torch.zeros((B, 1)).to(x.device), padding_mask), dim=1)
        
        x = self.embed_layer(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.norm(x)

        return self.head(x[:, 0])
    
if __name__ == "__main__":
        INPUT_DIM = 3
        SEQ_LEN = 1024
        NUM_CLASSES = 60
        EMBED_DIM = 8
        DEPTH = 4
        NUM_HEADS = 4
        MLP_RATIO = 4.
        QKV_BIAS = True
        DROP_RATE = 0.1
        ATTN_DROP_RATE = 0.1
        DROP_PATH_RATE = 0.
        NORM_LAYER = None
        ACT_LAYER = nn.GELU
        model = ClassificationTransformer(INPUT_DIM,
                                          SEQ_LEN,
                                          NUM_CLASSES,
                                          EMBED_DIM,
                                          DEPTH,
                                          NUM_HEADS,
                                          MLP_RATIO,
                                          QKV_BIAS,
                                          DROP_RATE,
                                          ATTN_DROP_RATE,
                                          DROP_PATH_RATE,
                                          NORM_LAYER,
                                          ACT_LAYER)
        
        BATCH_SIZE = 64
        SEQ_LEN = 1024
        x = torch.randn((BATCH_SIZE, SEQ_LEN, INPUT_DIM))
        y = model(x)
        print(y.shape)