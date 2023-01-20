# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

IMAGENET_DIR=""

# Both versions of MAE_ST at 10%
python src/image_mae_submitit_pretrain.py --job_dir="experiments/image_mae/train" \
    --nodes=8 --exclude="" \
    --partition=learnai4p --acount=all \
    --distributed \
    --epochs=800 --warmup_epochs=40 \
    --batch_size=64 --accum_iter=1 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio=0.75 \
    --blr=1.5e-4 --weight_decay=0.05 \
    --data_path=$IMAGENET_DIR