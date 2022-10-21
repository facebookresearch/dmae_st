# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Set modules for EFA reduce
source /data/home/alexnw/load_modules.sh

python src/pc3d_submitit_preprocess_kinetics.py --job_dir="experiments/pc3d/preprocess_kinetics/run_preprocess" --ngpus=8 --nodes=20 --exclude=60 \
    --name="K400 Data Preprocessing" \
    --data_save_dir="data/kinetics_400_annotations-videos" \
    --distributed
    