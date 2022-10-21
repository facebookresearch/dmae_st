# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh

# conda activate /data/home/alexnw/miniconda/envs/torch110

# python src/mae_submitit_pretrain.py --job_dir="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA" --ngpus=8 --nodes=8 --exclude=260,217,96,39,47,5,25,76 \
#     --epochs=200 --warmup_epochs=20 \
#     --batch_size=3 --repeat_aug=4 --accum_iter=1 \
#     --input_size=224  --num_frames=16 --mask_ratio=0.90 --mask_schedule=const --mask_type=st --norm_pix_loss \
#     --blr=4e-4 \
#     --num_workers=12 --pin_mem \
#     --model=mae_vit_large_patch16 \
#     --decoder_embed_dim=512 --decoder_depth=4 --decoder_num_heads=16 \
#     --t_patch_size=2 --pred_t_dim=8 --sampling_rate=4 \
#     --encoder_attn=AttentionWithCls --decoder_attn=AttentionOrg \
#     --clip_grad=0.02 --learnable_pos_embed --sep_pos_embed --cls_embed --fp32

python src/mae_submitit_pretrain.py --job_dir="experiments/mae-st/MAE_ST-75epochs-k400v1" \
    --ngpus=8 --nodes=8 --exclude="" \
    --partition=lowpri --account=all --timeout=1440 \
    --epochs=75 --warmup_epochs=20 \
    --batch_size=3 --repeat_aug=4 --accum_iter=1 \
    --input_size=224  --num_frames=16 --mask_ratio=0.90 --mask_schedule=const --mask_type=st --norm_pix_loss \
    --blr=4e-4 \
    --num_workers=12 --pin_mem \
    --model=mae_vit_large_patch16 \
    --decoder_embed_dim=512 --decoder_depth=4 --decoder_num_heads=16 \
    --t_patch_size=2 --pred_t_dim=8 --sampling_rate=4 \
    --encoder_attn=AttentionWithCls --decoder_attn=AttentionOrg \
    --clip_grad=0.02 --learnable_pos_embed --sep_pos_embed --cls_embed --fp32

python src/mae_submitit_pretrain.py --job_dir="experiments/mae-st/MAE_ST-75epochs-k400v2" \
    --ngpus=8 --nodes=8 --exclude="" \
    --partition=lowpri --account=all --timeout=1440 \
    --epochs=75 --warmup_epochs=20 \
    --batch_size=3 --repeat_aug=4 --accum_iter=1 \
    --input_size=224  --num_frames=16 --mask_ratio=0.90 --mask_schedule=const --mask_type=st --norm_pix_loss \
    --blr=4e-4 \
    --num_workers=12 --pin_mem \
    --model=mae_vit_large_patch16 \
    --decoder_embed_dim=512 --decoder_depth=4 --decoder_num_heads=16 \
    --t_patch_size=2 --pred_t_dim=8 --sampling_rate=4 \
    --encoder_attn=AttentionWithCls --decoder_attn=AttentionOrg \
    --clip_grad=0.02 --learnable_pos_embed --sep_pos_embed --cls_embed --fp32 --k400v2