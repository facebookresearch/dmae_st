# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# python src/output_incorrect_test_ex_submitit.py --job_dir=experiments/output/incorrect_test_ex \
#     --nodes=2 --partition=lowpri

python src/output_incorrect_test_ex_submitit.py --job_dir=experiments/output/incorrect_test_ex-barrier \
    --num_gpus=8 --partition=lowpri --exclude="219"