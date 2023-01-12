# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

python src/pc3d_submitit_preprocess_imagenet.py --job_dir="experiments/pc3d/preprocess_imagenet/run_preprocess_in1k" \
    --nodes=1 --num_gpus=1 --exclude="" \
    --partition=learnai4p --account=all \
    --data_dir="/datasets01/imagenet_full_size/061417/train" \
    --data_save_dir="data/imagenet_1000-annotations" \
    