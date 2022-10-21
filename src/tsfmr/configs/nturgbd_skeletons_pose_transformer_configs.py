from configs.configs import register_config


@register_config
def nturgbd_skeletons_pose_transformer():
    return {
        "eval_type": "xsub",
        "num_frames": 32,
        
        "batch_size": 16,
        
        "epochs": 200,
        "base_lr": 0.01,
        "lr_decay_min": 0.0,
        "lr_warmup_epochs": 1,
        
        "embed_dim": 16,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_rate": 0.,
        "attn_drop_rate": 0.,
        "drop_path_rate": 0.,
        "norm_layer": None,
        "act_layer": None,
        "hybrid_pos_embed": False,
    }
    
@register_config
def hybrid_pos_embed():
    return {
        "hybrid_pos_embed": True
    }