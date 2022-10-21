# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
source /data/home/alexnw/miniconda/etc/profile.d/conda.sh

# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh

conda activate /data/home/alexnw/miniconda/envs/torch110
cd /fsx/kart

python /data/home/alexnw/alexnw/projects/skel_act_recg/src/dmae_submitit_pretrain.py \
    --job_dir="experiments/dmae-st/DMAE_ST-200epochs-maskratio075-TempCos150_050" \
    --ngpus=8 --nodes=8 --exclude="" \
    --partition=hipri --account=all \
    --epochs=200 --warmup_epochs=20 \
    --batch_size=2 --repeat_aug=4 --accum_iter=1 \
    --input_size=224  --num_frames=16 --mask_ratio=0.75 --mask_schedule=const --mask_type=directed --norm_pix_loss \
    --temperature_schedule=cos --temperature_end=0.5 --temperature=1.5 \
    --blr=4e-4 \
    --num_workers=12 --pin_mem \
    --model=dmae_vit_large_patch16 \
    --decoder_embed_dim=512 --decoder_depth=4 --decoder_num_heads=16 \
    --t_patch_size=2 --pred_t_dim=8 --sampling_rate=4 \
    --encoder_attn=AttentionWithCls --decoder_attn=AttentionOrg \
    --clip_grad=0.02 --learnable_pos_embed --sep_pos_embed --cls_embed --fp32