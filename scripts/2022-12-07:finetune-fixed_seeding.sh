# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Both versions of MAE_ST at 10%
python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/finetune-fixed_frac010_sample" \
    --nodes=4 --ngpus=8 --exclude="" \
    --partition=lowpri --account=all \
    --distributed \
    --epochs=100 --warmup_epochs=10 --dist_eval --checkpoint_period=10 \
    --batch_size=3 --repeat_aug=1 --accum_iter=1 \
    --input_size=224 --num_frames=16 \
    --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
    --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
    --num_workers=12 --pin_mem \
    --blr=2.4e-3 \
    --model=vit_large_patch16 \
    --t_patch_size=2 --sampling_rate=4 \
    --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
    --encoder_attn=AttentionWithCls \
    --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
    --finetune_fraction=0.1 \
    --finetune="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/checkpoint-00199.pth"

# This is the second version with K400v2 -- idr how they're diff.
python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-fixed_frac010_sample" \
    --nodes=4 --ngpus=8 --exclude="" \
    --partition=lowpri --account=all \
    --distributed \
    --epochs=100 --warmup_epochs=10 --dist_eval --checkpoint_period=10 \
    --batch_size=3 --repeat_aug=1 --accum_iter=1 \
    --input_size=224 --num_frames=16 \
    --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
    --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
    --num_workers=12 --pin_mem \
    --blr=2.4e-3 \
    --model=vit_large_patch16 \
    --t_patch_size=2 --sampling_rate=4 \
    --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
    --encoder_attn=AttentionWithCls \
    --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
    --finetune_fraction=0.1 \
    --finetune="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/checkpoints/checkpoint-00199.pth"

# This should be the curriculum model that got the best results.
python src/mae_submitit_finetune.py --job_dir="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/finetune-fixed_frac010_sample" \
    --ngpus=8 --nodes=4 --exclude="" \
    --partition=lowpri --account=all \
    --distributed \
    --epochs=100 --warmup_epochs=10 --dist_eval --checkpoint_period=10 \
    --batch_size=3 --repeat_aug=1 --accum_iter=1 \
    --input_size=224 --num_frames=16 \
    --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
    --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
    --num_workers=12 --pin_mem \
    --blr=2.4e-3 \
    --model=vit_large_patch16 \
    --t_patch_size=2 --sampling_rate=4 \
    --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
    --encoder_attn=AttentionWithCls \
    --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
    --finetune_fraction=0.1 \
    --finetune="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/checkpoint-00199.pth"

# This is the first half of the sweep results.
for TEMP in -1e10 -2e0 -1e0 -0.5e0 0; do
    python src/mae_submitit_finetune.py --job_dir="experiments/dmae-st/DMAE_ST-full-sweep/DMAE_ST-full-TEMP$TEMP/finetune-fixed_frac010_sample" \
        --ngpus=8 --nodes=4 --exclude="" \
        --partition=lowpri --account=all \
        --distributed \
        --epochs=100 --warmup_epochs=10 --dist_eval --checkpoint_period=10 \
        --batch_size=3 --repeat_aug=1 --accum_iter=1 \
        --input_size=224 --num_frames=16 \
        --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
        --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
        --num_workers=12 --pin_mem \
        --blr=2.4e-3 \
        --model=vit_large_patch16 \
        --t_patch_size=2 --sampling_rate=4 \
        --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
        --encoder_attn=AttentionWithCls \
        --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
        --finetune_fraction=0.1 \
        --finetune="experiments/dmae-st/DMAE_ST-full-sweep/DMAE_ST-full-TEMP$TEMP/checkpoints/checkpoint-00199.pth" # extra /checkpoints/ folder
done