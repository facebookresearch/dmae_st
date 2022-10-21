# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh
python src/mae_submitit_finetune.py --job_dir="experiments/dmae-st/8x8A100-DMAE_ST-default/finetune" \
    --ngpus=8 --nodes=8 --exclude="19" \
    --distributed \
    --epochs=100 --warmup_epochs=10 --dist_eval \
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
    --finetune="experiments/dmae-st/8x8A100-DMAE_ST-default/checkpoint-00199.pth"