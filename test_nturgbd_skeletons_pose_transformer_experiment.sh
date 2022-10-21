# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
rm -rf experiments/nturgbd_skeletons_pose_transformer/TEST
python src/nturgbd_skeletons_pose_transformer_train.py --dir=experiments/nturgbd_skeletons_pose_transformer/TEST --data_dir=data/nturgb+d_skeletons --config=default --dist