# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh

# conda activate /data/home/alexnw/miniconda/envs/torch110

# python src/dmae_submitit_pretrain.py --job_dir="experiments/dmae-st/8x8A100-DMAE_ST-default" --ngpus=8 --nodes=8 --exclude="19" \
#     --partition=hipri --account=all \
#     --epochs=200 --warmup_epochs=20 \
#     --batch_size=3 --repeat_aug=4 --accum_iter=1 \
#     --input_size=224  --num_frames=16 --mask_ratio=0.90 --mask_schedule=const --mask_type=directed --norm_pix_loss \
#     --blr=4e-4 \
#     --num_workers=12 --pin_mem \
#     --model=dmae_vit_large_patch16 \
#     --decoder_embed_dim=512 --decoder_depth=4 --decoder_num_heads=16 \
#     --t_patch_size=2 --pred_t_dim=8 --sampling_rate=4 \
#     --encoder_attn=AttentionWithCls --decoder_attn=AttentionOrg \
#     --clip_grad=0.02 --learnable_pos_embed --sep_pos_embed --cls_embed --fp32

# python src/dmae_submitit_pretrain.py --job_dir="experiments/dmae-st/DMAE_ST-75epochs" \
#     --ngpus=8 --nodes=8 --exclude="147" \
#     --partition=lowpri --account=all --timeout=1440 \
#     --epochs=75 --warmup_epochs=20 \
#     --batch_size=3 --repeat_aug=4 --accum_iter=1 \
#     --input_size=224  --num_frames=16 --mask_ratio=0.90 --mask_schedule=const --mask_type=directed --norm_pix_loss \
#     --temperature=1.0 \
#     --blr=4e-4 \
#     --num_workers=12 --pin_mem \
#     --model=dmae_vit_large_patch16 \
#     --decoder_embed_dim=512 --decoder_depth=4 --decoder_num_heads=16 \
#     --t_patch_size=2 --pred_t_dim=8 --sampling_rate=4 \
#     --encoder_attn=AttentionWithCls --decoder_attn=AttentionOrg \
#     --clip_grad=0.02 --learnable_pos_embed --sep_pos_embed --cls_embed --fp32

python src/dmae_submitit_pretrain.py --job_dir="experiments/dmae-st/DMAE_ST-200epochs-curriculum-flip_mask" \
    --nodes=8 --exclude="" \
    --partition=hipri --account=all \
    --epochs=200 --warmup_epochs=20 \
    --batch_size=3 --repeat_aug=4 --accum_iter=1 \
    --input_size=224  --num_frames=16 --mask_ratio=0.90 --mask_schedule=const --mask_type=directed --norm_pix_loss \
    --temperature_schedule=cos --temperature_end=0.5 --temperature=1.5 \
    --flip_mask \
    --blr=4e-4 \
    --num_workers=12 --pin_mem \
    --model=dmae_vit_large_patch16 \
    --decoder_embed_dim=512 --decoder_depth=4 --decoder_num_heads=16 \
    --t_patch_size=2 --pred_t_dim=8 --sampling_rate=4 \
    --encoder_attn=AttentionWithCls --decoder_attn=AttentionOrg \
    --clip_grad=0.02 --learnable_pos_embed --sep_pos_embed --cls_embed --fp32