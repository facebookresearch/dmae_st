"""
Based on 06-13:0 and 06-14:0, models are overfitting considerably. Though they still do not get great results
in any regard (SotA is around 94%), they overfit to ~70% while evaluating at 50-60%. 
Try DROPOUT as the starting approach.

Lets just run dropout on everything
"""

port=40000
for base_lr in 0.05; do
    for depth in 6; do
        for dropout in 0.05 0.1; do
            for eval_type in "xsub"; do
                bash scripts/launch_scripts/run_4gpu.sh nturgbd_skeletons_transformer_train.py 1:with_dropout/train-EvalType$eval_type-BASELR$base_lr-DEPTH$depth-DROPOUTS$dropout "--data_dir=data/nturgb+d_skeletons --config=default --port=$port --eval_type=$eval_type --base_lr=$base_lr --depth=$depth --drop_rate=$dropout --attn_drop_rate=$dropout"
                port=$((port+1))
                bash scripts/launch_scripts/run_4gpu.sh nturgbd_skeletons_pose_transformer_train.py 1:with_dropout/train-HybridPosEmbed-EvalType$eval_type-BASELR$base_lr-DEPTH$depth-DROPOUTS$dropout "--data_dir=data/nturgb+d_skeletons --config=default.hybrid_pos_embed --port=$port --eval_type=$eval_type --base_lr=$base_lr --depth=$depth --drop_rate=$dropout --attn_drop_rate=$dropout"
                port=$((port+1))
            done
        done
    done
done