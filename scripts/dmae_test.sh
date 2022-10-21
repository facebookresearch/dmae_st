# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh
python src/mae_submitit_test.py --job_dir="experiments/dmae-st/8x8A100-DMAE_ST-default/test" --ngpus=8 --nodes=8 --exclude="" \
    --partition=lowpri --timeout=1440 \
    --distributed \
    --dist_eval \
    --batch_size=64 \
    --input_size=224 --num_frames=16 \
    --num_workers=12 --pin_mem \
    --model=vit_large_patch16 \
    --t_patch_size=2 --sampling_rate=4 \
    --dropout=0.3 \
    --encoder_attn=AttentionWithCls \
    --cls_embed --sep_pos_embed --fp32 \
    --finetune="experiments/dmae-st/8x8A100-DMAE_ST-default/finetune/checkpoint-00099.pth"