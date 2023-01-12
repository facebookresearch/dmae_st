# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

python src/pc3d_submitit_preprocess_imagenet.py --job_dir="experiments/pc3d/preprocess_imagenet/run_preprocess" \
    --nodes=1 --ngpus=1 --exclude="" \
    --partition=lowpri --account=all \
    --name="IN1000 BBox Data Preprocessing" \
    --data_dir = ${IN1000 DIRECTORY HERE} \
    --data_save_dir="data/imagenet_1000-annotations" \
    