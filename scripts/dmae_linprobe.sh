# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
source /data/home/alexnw/load_modules.sh

python src/mae_submitit_linprobe.py --job_dir="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/linprobe-frac100" \
    --nodes=6 --exclude="" \
    --partition=lowpri --account=all \
    --distributed \
    --epochs=10 --warmup_epochs=1 --dist_eval \
    --batch_size=8 --repeat_aug=10 --accum_iter=1 \
    --input_size=224 --num_frames=16 \
    --smoothing=0.1 \
    --num_workers=12 --pin_mem \
    --blr=0.1 \
    --model=vit_large_patch16 \
    --t_patch_size=2 --sampling_rate=4 \
    --encoder_attn=AttentionWithCls \
    --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
    --finetune_fraction=1.0 \
    --finetune="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/checkpoint-00199.pth"

python src/mae_submitit_linprobe.py --job_dir="experiments/dmae-st/DMAE_ST-200epochs-curriculum-flip_mask/linprobe-frac100" \
    --nodes=6 --exclude="" \
    --partition=lowpri --account=all \
    --distributed \
    --epochs=10 --warmup_epochs=1 --dist_eval \
    --batch_size=8 --repeat_aug=10 --accum_iter=1 \
    --input_size=224 --num_frames=16 \
    --smoothing=0.1 \
    --num_workers=12 --pin_mem \
    --blr=0.1 \
    --model=vit_large_patch16 \
    --t_patch_size=2 --sampling_rate=4 \
    --encoder_attn=AttentionWithCls \
    --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
    --finetune_fraction=1.0 \
    --finetune="experiments/dmae-st/DMAE_ST-200epochs-curriculum-flip_mask/checkpoints/checkpoint-00199.pth"
    
python src/mae_submitit_linprobe.py --job_dir="experiments/dmae-st/8x8A100-DMAE_ST-default/linprobe-frac100" \
    --nodes=6 --exclude="" \
    --partition=lowpri --account=all \
    --distributed \
    --epochs=10 --warmup_epochs=1 --dist_eval \
    --batch_size=8 --repeat_aug=10 --accum_iter=1 \
    --input_size=224 --num_frames=16 \
    --smoothing=0.1 \
    --num_workers=12 --pin_mem \
    --blr=0.1 \
    --model=vit_large_patch16 \
    --t_patch_size=2 --sampling_rate=4 \
    --encoder_attn=AttentionWithCls \
    --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
    --finetune_fraction=1.0 \
    --finetune="experiments/dmae-st/8x8A100-DMAE_ST-default/checkpoint-00199.pth"
    