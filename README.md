# Skeleton-based action recognition

Repository for learning pose-aware video representations for downstream tasks.

## Respository structure
Code is stored in `src/` and all runnable scripts are in this directory. This is an in-development repository and will contain multiple different experiments. The respective scripts for each type of experiment will be differentiated by the first word in the script. For example, all experiments related to masked autoencoders (mae) will be named `mae_{script_name}.py`. 

Bash scripts that contain settings for each of these scripts of kept in `scripts/`. Some may be run python files that will submit SLURM jobs using submitit, some will call a script in `scripts/launch_scripts` which will submit the SLURM job.

Experiments will create and store all outputs in a new directory `experiments/`. 

## Current experiment types

### (1) Transformer baseline action prediction (_not under development_)
Prefix: `tsfmr_` \
Experiment files: `src/tsfmr` \

Code is written from scratch except for transformer model implementation which is borrowed from [minGPT](https://github.com/karpathy/minGPT/tree/master/mingpt). 

Predicts action labels given 3D pose information on [NTURGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/).

### (2) Masked autoencoders for video representations
Prefix: `mae_` \
Experiment files: `src/mae`

Code is adapted from [D36893220](https://www.internalfb.com/diff/D36893220) to run on AWS cluster and run without internal build tools. 

Pretrains a representation used masked-autoencoders on video data that will then be used for action classification on the [K400](https://github.com/cvdfoundation/kinetics-dataset) dataset.

How to run: 
```
bash scripts/mae_pretrain.sh
```

### (3) PoseConv3D
Prefix: `pc3d_` \
Experiment files: `src/pc3d`

Ideas taken from the [PoseConv3D](https://arxiv.org/abs/2104.13586) paper. Currently, only a data processing procedure where 
bounding boxes are extracted from a video dataset.

### (4) Directed Masked Autoencoders
Prefix: `dmae_` \
Experiment files: `src/dmae`

All new files, models, experiments and utilities that enable directed masking for a masked autoencoder approach for video pretraining.

## Environment setup
Packages are all provided in `requirements.txt`. 

### Notes
1. `torchvision` needs to be build from source with `ffmpeg` backend to properly decode video data for `Masked autoencoder` data loading. Follow instructions [here](https://github.com/pytorch/vision) in the section regarding `Video Backend`. 
