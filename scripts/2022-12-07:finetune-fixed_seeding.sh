# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

OUT_DIR="experiments/dmae-st/finetune_10pct"

# # Both versions of MAE_ST at 10%
# python src/mae_submitit_finetune.py --job_dir="$OUT_DIR/8x8A100-MAE_ST-bs3-fp32-EFA" \
#     --nodes=4 --exclude="" \
#     --partition=learnai4p --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval --checkpoint_period=10 \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=0.1 \
#     --finetune="/fsx/alexnw/experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/checkpoint-00199.pth"

# # This is the second version with K400v2 -- idr how they're diff.
# python src/mae_submitit_finetune.py --job_dir="$OUT_DIR/MAE_ST-200epochs-k400v2-288px" \
#     --nodes=4 --exclude="" \
#     --partition=learnai4p --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval --checkpoint_period=10 \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=0.1 \
#     --finetune="/fsx/alexnw/experiments/mae-st/MAE_ST-200epochs-k400v2-288px/checkpoints/checkpoint-00199.pth"

# # This should be the curriculum model that got the best results.
# python src/mae_submitit_finetune.py --job_dir="$OUT_DIR/DMAE_ST-200epochs-TempCos150_050" \
#     --nodes=4 --exclude="" \
#     --partition=learnai4p --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval --checkpoint_period=10 \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=0.1 \
#     --finetune="/fsx/alexnw/experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/checkpoint-00199.pth"

# This is the first half of the sweep results.
# for TEMP in -1e10 -2e0 -1e0 -0.5e0 0; do
for TEMP in 1.5e0; do
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH python src/mae_submitit_finetune.py --job_dir="$OUT_DIR/DMAE_ST-full-TEMP$TEMP" \
        --nodes=4 --exclude="" \
        --partition=learnai4p --account=all \
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
        --finetune="/fsx/kart/skel_act_recg/experiments/dmae-st/DMAE_ST-full-sweep-positives/DMAE_ST-full-TEMP$TEMP/checkpoints/checkpoint-00199.pth" # extra /checkpoints/ folder
done