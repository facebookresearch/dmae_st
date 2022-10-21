# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh
# python src/mae_submitit_test.py --job_dir="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/test-val_and_test" \
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
#     --finetune="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/finetune/checkpoint-00095.pth"

# python src/mae_submitit_test.py --job_dir="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/test-frac010-val_and_test" \
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
#     --finetune="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/finetune-frac010/checkpoints/checkpoint-00095.pth"

# python src/mae_submitit_test.py --job_dir="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/test-epoch75-frac010-val_and_test" \
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
#     --finetune="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/finetune-epoch75-frac010/checkpoint-00098.pth"

# python src/mae_submitit_test.py --job_dir="experiments/mae-st/MAE_ST-75epochs-k400v2/test-frac010-val_and_test" \
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
#     --finetune="experiments/mae-st/MAE_ST-75epochs-k400v2/finetune-frac010/checkpoints/checkpoint-00096.pth"

# python src/mae_submitit_test.py --job_dir="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/test-frac010-val_and_test" \
#     --nodes=2 --exclude="" \
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
#     --finetune="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-frac010/checkpoints/checkpoint-00095.pth"

# python src/mae_submitit_test.py --job_dir="experiments/mae-st/MAE_ST-75epochs-k400v1/test-frac010-val_and_test" \
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
#     --finetune="experiments/mae-st/MAE_ST-75epochs-k400v1/finetune-frac010/checkpoints/checkpoint-00095.pth"
    
python src/mae_submitit_test.py --job_dir="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/test-frac100-val_and_test" \
    --nodes=8 --exclude="265" \
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
    --finetune="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-frac100/checkpoints/checkpoint-00090.pth"

# python src/mae_submitit_test.py --job_dir="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/test-frac010-epochs1e3" \
#     --nodes=6 --exclude="" \
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
#     --finetune="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-frac010-epochs1e3/checkpoints/checkpoint-00910.pth"