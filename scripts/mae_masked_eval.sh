source /data/home/alexnw/load_modules.sh

python src/dmae_submitit_masked_finetune.py --job_dir="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/masked_eval" \
    --nodes=6 --exclude="" \
    --partition=lowpri --account=all \
    --distributed \
    --epochs=100 --warmup_epochs=10 --dist_eval \
    --batch_size=3 --repeat_aug=1 --accum_iter=1 \
    --input_size=224 --num_frames=16 \
    --num_workers=12 --pin_mem \
    --t_patch_size=2 --sampling_rate=4 \
    --dropout=0.3 --layer_decay=0.75 --drop_path_rate=0.2 --rel_pos_init_std=0.02 \
    --encoder_attn=AttentionWithCls \
    --cls_embed --sep_pos_embed --clip_grad=5.0 --fp32 \
    --finetune="experiments/mae-st/MAE_ST-200epochs-k400v2-288px/finetune-frac100/checkpoints/checkpoint-00090.pth" \
    --eval