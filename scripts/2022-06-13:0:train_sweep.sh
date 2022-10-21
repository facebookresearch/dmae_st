"""
Initial sweep of training a batch of experiments using pose and regular transformers.
Memory is a significant issue, had to use batch_size=16 in order to make it even fit on an A100...
Only swept a few minor params.
"""

port=40000
for base_lr in 0.05 0.1; do
    for depth in 4 6; do
        for num_heads in 4; do
            for eval_type in "xsub"; do
                bash scripts/launch_scripts/run_4gpu.sh nturgbd_skeletons_pose_transformer_train.py train-EvalType$eval_type-BASELR$base_lr-DEPTH$depth-NUMHEADS$num_heads "--data_dir=data/nturgb+d_skeletons --config=default --port=$port --eval_type=$eval_type --base_lr=$base_lr --depth=$depth --num_heads=$num_heads"
                port=$((port+1))
                bash scripts/launch_scripts/run_4gpu.sh nturgbd_skeletons_transformer_train.py train-EvalType$eval_type-BASELR$base_lr-DEPTH$depth-NUMHEADS$num_heads "--data_dir=data/nturgb+d_skeletons --config=default --port=$port --eval_type=$eval_type --base_lr=$base_lr --depth=$depth --num_heads=$num_heads"
                port=$((port+1))
            done;
        done;
    done;
done;