# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh

# conda activate /data/home/alexnw/miniconda/envs/torch110

python src/dmae_submitit_maskratios.py --job_dir="experiments/dmae-st/4x8A100-MASKRATIOS" --ngpus=8 --nodes=8 --exclude="19" \
    --partition=hipri --account=all \
    --epochs=1 \
    --batch_size=32 --repeat_aug=8 \
    --input_size=224  --num_frames=16 --mask_ratio=0.90 --mask_schedule=const --mask_type=directed \
    --num_workers=12 --pin_mem \
    --model=dmae_vit_large_patch16 \
    --decoder_embed_dim=512 --decoder_depth=4 --decoder_num_heads=16 \
    --t_patch_size=2 --pred_t_dim=8 --sampling_rate=4 \
    --encoder_attn=AttentionWithCls --decoder_attn=AttentionOrg \
    --clip_grad=0.02 --learnable_pos_embed --sep_pos_embed --cls_embed --fp32