# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
source /data/home/alexnw/miniconda/etc/profile.d/conda.sh

# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh

conda activate /data/home/alexnw/miniconda/envs/torch110
cd /fsx/kart


for SEED in 1 2 3 4 5; do
    python src/mae_submitit_full.py --job_dir="experiments/mae-st/MAE_ST-full-sweep-Seed/MAE_ST-full-SEED$SEED" \
        --nodes=8 --exclude="134,135,260,50,219" \
        --partition=hipri --account=all \
        --PRETRAIN.epochs=1 --PRETRAIN.warmup_epochs=20 \
        --PRETRAIN.checkpoint_period=20 \
        --PRETRAIN.batch_size=3 --PRETRAIN.repeat_aug=4 --PRETRAIN.accum_iter=1 \
        --PRETRAIN.input_size=224  --PRETRAIN.num_frames=16 --PRETRAIN.mask_ratio=0.90 --PRETRAIN.mask_schedule=const --PRETRAIN.mask_type=directed --PRETRAIN.norm_pix_loss \
        --PRETRAIN.blr=4e-4 \
        --PRETRAIN.num_workers=12 --PRETRAIN.pin_mem \
        --PRETRAIN.model=dmae_vit_large_patch16_v2 \
        --PRETRAIN.decoder_embed_dim=512 --PRETRAIN.decoder_depth=4 --PRETRAIN.decoder_num_heads=16 \
        --PRETRAIN.t_patch_size=2 --PRETRAIN.pred_t_dim=8 --PRETRAIN.sampling_rate=4 \
        --PRETRAIN.encoder_attn=AttentionWithCls --PRETRAIN.decoder_attn=AttentionOrg \
        --PRETRAIN.clip_grad=0.02 --PRETRAIN.learnable_pos_embed --PRETRAIN.sep_pos_embed --PRETRAIN.cls_embed --PRETRAIN.fp32 \
        --PRETRAIN.seed=$SEED \
        --FINETUNE.distributed \
        --FINETUNE.epochs=100 --FINETUNE.warmup_epochs=10 --FINETUNE.dist_eval \
        --FINETUNE.checkpoint_period=5 \
        --FINETUNE.batch_size=3 --FINETUNE.repeat_aug=1 --FINETUNE.accum_iter=1 \
        --FINETUNE.input_size=224 --FINETUNE.num_frames=16 \
        --FINETUNE.smoothing=0.1 --FINETUNE.aa=rand-m7-mstd0.5-inc1 \
        --FINETUNE.rand_aug --FINETUNE.mixup=0.8 --FINETUNE.cutmix=1.0 --FINETUNE.mixup_prob=1.0 \
        --FINETUNE.num_workers=12 --FINETUNE.pin_mem \
        --FINETUNE.blr=2.4e-3 \
        --FINETUNE.model=vit_large_patch16 \
        --FINETUNE.t_patch_size=2 --FINETUNE.sampling_rate=4 \
        --FINETUNE.dropout=0.3 --FINETUNE.layer_decay=0.75 --FINETUNE.drop_path_rate=0.2 --FINETUNE.rel_pos_init_std=0.02 \
        --FINETUNE.encoder_attn=AttentionWithCls \
        --FINETUNE.cls_embed --FINETUNE.sep_pos_embed --FINETUNE.clip_grad=5.0 --FINETUNE.fp32 \
        --FINETUNE.finetune_fraction=1.0 \
        --TEST.distributed \
        --TEST.dist_eval \
        --TEST.batch_size=64 \
        --TEST.input_size=224 --TEST.num_frames=16 \
        --TEST.num_workers=12 --TEST.pin_mem \
        --TEST.model=vit_large_patch16 \
        --TEST.t_patch_size=2 --TEST.sampling_rate=4 \
        --TEST.dropout=0.3 \
        --TEST.encoder_attn=AttentionWithCls \
        --TEST.cls_embed --TEST.sep_pos_embed --TEST.fp32
done