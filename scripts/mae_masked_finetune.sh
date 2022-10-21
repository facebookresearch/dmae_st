# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
source /data/home/alexnw/load_modules.sh

python src/dmae_submitit_masked_finetune.py --job_dir="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/masked-finetune-frac100" \
    --nodes=8 --exclude="" \
    --partition=lowpri --account=all \
    --distributed \
    --epochs=100 --warmup_epochs=10 --dist_eval \
    --batch_size=3 --repeat_aug=1 --accum_iter=1 \
    --input_size=224 --num_frames=16 \
    --smoothing=0.1 \
    --num_workers=12 --pin_mem \
    --blr=2.4e-3 \
    --model=masked_vit_large_patch16 \
    --t_patch_size=2 --sampling_rate=4 \
    --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
    --encoder_attn=AttentionWithCls \
    --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
    --finetune_fraction=1.0 \
    --finetune="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/checkpoints/checkpoint-00199.pth"