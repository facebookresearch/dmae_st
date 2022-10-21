# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""
Pose transformer didn't outperform the learnable representation.
What if we could just combine the two methods and use it as a jumpstart but also give it the learnable option?
"""

port=40000
for base_lr in 0.05; do
    for depth in 6 8; do
        for num_heads in 4; do
            for eval_type in "xsub"; do
                bash scripts/launch_scripts/run_4gpu.sh nturgbd_skeletons_pose_transformer_train.py with_hybrids/train-EvalType$eval_type-BASELR$base_lr-DEPTH$depth-NUMHEADS$num_heads "--data_dir=data/nturgb+d_skeletons --config=default --port=$port --eval_type=$eval_type --base_lr=$base_lr --depth=$depth --num_heads=$num_heads"
                port=$((port+1))
                bash scripts/launch_scripts/run_4gpu.sh nturgbd_skeletons_pose_transformer_train.py with_hybrids/train-EvalType$eval_type-BASELR$base_lr-DEPTH$depth-NUMHEADS$num_heads-hybrid "--data_dir=data/nturgb+d_skeletons --config=default.hybrid_pos_embed --port=$port --eval_type=$eval_type --base_lr=$base_lr --depth=$depth --num_heads=$num_heads"
                port=$((port+1))
            done;
        done;
    done;
done;