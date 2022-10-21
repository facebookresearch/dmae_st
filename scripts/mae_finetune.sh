# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh
# python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/finetune" --ngpus=8 --nodes=8 --exclude=260,217,96,39,47,76,72 \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/checkpoint-00199.pth"

# python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/finetune-frac010" \
#     --ngpus=8 --nodes=4 --exclude="50" \
#     --partition=hipri --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=0.1 \
#     --finetune="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/checkpoint-00199.pth"

# python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/finetune-epoch75-frac010" \
#     --ngpus=8 --nodes=4 --exclude="50" \
#     --partition=hipri --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=0.1 \
#     --finetune="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/checkpoint-00074.pth"

# python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/finetune-epoch75-frac100" \
#     --ngpus=8 --nodes=4 --exclude="50" \
#     --partition=hipri --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=1.0 \
#     --finetune="experiments/mae-st/8x8A100-MAE_ST-bs3-fp32-EFA/checkpoint-00074.pth"

# python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/MAE_ST-75epochs-k400v2/finetune-frac010" \
#     --ngpus=8 --nodes=4 --exclude="50" \
#     --partition=hipri --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=0.1 \
#     --finetune="experiments/mae-st/MAE_ST-75epochs-k400v2/checkpoint-00074.pth"

# python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/MAE_ST-75epochs-k400v2/finetune-frac100" \
#     --ngpus=8 --nodes=4 --exclude="50" \
#     --partition=hipri --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=1.0 \
#     --finetune="experiments/mae-st/MAE_ST-75epochs-k400v2/checkpoint-00074.pth"

# python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-frac100" \
#     --nodes=6 --exclude="" \
#     --partition=hipri --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=1.0 \
#     --finetune="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/checkpoints/checkpoint-00199.pth"

# python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-frac010" \
#     --nodes=2 --exclude="" \
#     --partition=hipri --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=0.1 \
#     --finetune="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/checkpoints/checkpoint-00199.pth"

# python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/MAE_ST-75epochs-k400v1/finetune-frac010" \
#     --num_gpus=12 --exclude="" \
#     --partition=lowpri --account=all \
#     --distributed \
#     --epochs=100 --warmup_epochs=10 --dist_eval \
#     --batch_size=3 --repeat_aug=1 --accum_iter=1 \
#     --input_size=224 --num_frames=16 \
#     --smoothing=0.1 --aa=rand-m7-mstd0.5-inc1 \
#     --rand_aug --mixup=0.8 --cutmix=1.0 --mixup_prob=1.0 \
#     --num_workers=12 --pin_mem \
#     --blr=2.4e-3 \
#     --model=vit_large_patch16 \
#     --t_patch_size=2 --sampling_rate=4 \
#     --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
#     --encoder_attn=AttentionWithCls \
#     --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
#     --finetune_fraction=0.1 \
#     --finetune="experiments/mae-st/MAE_ST-75epochs-k400v1/checkpoints/checkpoint-00074.pth"

python src/mae_submitit_finetune.py --job_dir="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-frac010-epochs1e3" \
    --nodes=8 --exclude="265,159,158" \
    --partition=lowpri --account=all \
    --distributed \
    --epochs=1000 --warmup_epochs=10 --dist_eval --eval_period=10 \
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
    --finetune_fraction=0.1 \
    --finetune="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/checkpoints/checkpoint-00199.pth"