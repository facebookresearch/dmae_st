# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh
# python src/mae_submitit_test.py --job_dir="experiments/dmae-st/8x8A100-DMAE_ST-default/test-val_and_test" \
#     --num_gpus=16 --exclude="" \
#     --partition=lowpri --account=all \
#     --distributed \
#     --dist_eval \
#     --batch_size=64 \
#     --input_size=224 --num_frames=16 \
#     --num_workers=12 --pin_mem \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --fp32 \
#     --finetune="experiments/dmae-st/8x8A100-DMAE_ST-default/finetune/checkpoint-00089.pth"

# python src/mae_submitit_test.py --job_dir="experiments/dmae-st/8x8A100-DMAE_ST-default/test-frac010-val_and_test" \
#     --num_gpus=16 --exclude="77" \
#     --partition=lowpri --account=all \
#     --distributed \
#     --dist_eval \
#     --batch_size=64 \
#     --input_size=224 --num_frames=16 \
#     --num_workers=12 --pin_mem \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --fp32 \
#     --finetune="experiments/dmae-st/8x8A100-DMAE_ST-default/finetune-frac010/checkpoints/checkpoint-00095.pth"

# python src/mae_submitit_test.py --job_dir="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/test-frac010-val_and_test" \
#     --num_gpus=16 --exclude="77,152" \
#     --partition=lowpri --account=all \
#     --distributed \
#     --dist_eval \
#     --batch_size=64 \
#     --input_size=224 --num_frames=16 \
#     --num_workers=12 --pin_mem \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --fp32 \
#     --finetune="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/finetune-frac010/checkpoints/checkpoint-00098.pth"

# python src/mae_submitit_test.py --job_dir="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/test-frac100-val_and_test" \
#     --nodes=4 --exclude="278" \
#     --partition=lowpri --account=all \
#     --distributed \
#     --dist_eval \
#     --batch_size=64 \
#     --input_size=224 --num_frames=16 \
#     --num_workers=12 --pin_mem \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --fp32 \
#     --finetune="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/finetune-frac100/checkpoints/checkpoint-00099.pth"

# python src/mae_submitit_test.py --job_dir="experiments/dmae-st/DMAE_ST-75epochs/test-frac010-val_and_test" \
#     --num_gpus=16 --exclude="77,152" \
#     --partition=lowpri --account=all \
#     --distributed \
#     --dist_eval \
#     --batch_size=64 \
#     --input_size=224 --num_frames=16 \
#     --num_workers=12 --pin_mem \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --fp32 \
#     --finetune="experiments/dmae-st/DMAE_ST-75epochs/finetune-frac010/checkpoints/checkpoint-00095.pth"

# python src/mae_submitit_test.py --job_dir="experiments/dmae-st/DMAE_ST-200epochs-curriculum-flip_mask/test-frac100" \
#     --nodes=4 --exclude="278" \
#     --partition=hipri --account=all \
#     --distributed \
#     --dist_eval \
#     --batch_size=64 \
#     --input_size=224 --num_frames=16 \
#     --num_workers=12 --pin_mem \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --fp32 \
#     --finetune="experiments/dmae-st/DMAE_ST-200epochs-curriculum-flip_mask/finetune-frac100/checkpoints/checkpoint-00089.pth"

python src/mae_submitit_test.py --job_dir="experiments/dmae-st/DMAE_ST-200epochs-curriculum-flip_mask/test-frac010" \
    --nodes=6 --exclude="158,189" \
    --partition=lowpri --account=all \
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
    --finetune="experiments/dmae-st/DMAE_ST-200epochs-curriculum-flip_mask/finetune-frac010/checkpoints/checkpoint-00096.pth"

# python src/mae_submitit_test.py --job_dir="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/test-frac010-epochs1e3" \
#     --nodes=6 --exclude="63" \
#     --partition=lowpri --account=all \
#     --distributed \
#     --dist_eval \
#     --batch_size=64 \
#     --input_size=224 --num_frames=16 \
#     --num_workers=12 --pin_mem \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \mys
#     --dropout=0.3 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --fp32 \
#     --finetune="experiments/dmae-st/DMAE_ST-200epochs-TempCos150_050/finetune-frac010-epochs1e3/checkpoints/checkpoint-00800.pth"