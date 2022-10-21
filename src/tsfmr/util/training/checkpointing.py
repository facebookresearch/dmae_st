# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import os
import shutil

import torch
from torch import distributed as dist


def checkpoint_and_load(ckpt_dir, limit=10, **kwargs):
    def ckpt_fn(step):
        save_dict = {
            "step": step,
            "rng_state": torch.get_rng_state(),
            **{key: value.state_dict() for key, value in kwargs.items()}
        }
        
        ckpts = sorted(filter(lambda x: "ckpt" in x, os.listdir(ckpt_dir)),
                       key=lambda x: int(x[5:].split('.')[0]))
        if len(ckpts) == limit:
            os.remove(os.path.join(ckpt_dir, ckpts[0]))
        
        torch.save(save_dict,
                   os.path.join(ckpt_dir, "ckpt_{:04d}.pth.tar".format(step)))

    latest_ckpt = get_latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        print("Loading checkpoint: {}".format(latest_ckpt))
        step = load_checkpoint(latest_ckpt, **kwargs)
    else:
        print("No checkpoint found. Training from scratch.")
        step = 0

    return step, ckpt_fn

def load_checkpoint(checkpoint, **kwargs):
    if dist.is_initialized():
        load_dict = torch.load(checkpoint, map_location={"cuda:0":"cuda:{}".format(dist.get_rank() % torch.cuda.device_count())})
    else:
        load_dict = torch.load(checkpoint)

    
    torch.set_rng_state(load_dict['rng_state'])
    for key, value in kwargs.items():
        msg = value.load_state_dict(load_dict[key])
        print(f"Loading {key} with msg: {msg}")

    return load_dict['step']

def get_latest_checkpoint(ckpt_dir: str):
    try:
        latest_ckpt = sorted(filter(lambda x: "ckpt" in x, os.listdir(ckpt_dir)),
                            reverse=True,
                            key=lambda x: int(x[5:].split('.')[0]))[0]
        return os.path.join(ckpt_dir, latest_ckpt)
    except:
        return None


def move_newest_checkpoint(ckpt_dir: str, target_dir: str, num_checkpoints: int=1):
    """
    Moves the most recent checkpoint from ckpt_dir to target path.

    Args:
        ckpt_dir (str): location of saved checkpoint
        target_dir (str): new location
        num_checkpoints (int): number of checkpoints to copy
    """
    parent_dir = os.path.dirname(target_dir)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    for file in sorted(filter(lambda x: "ckpt" in x, os.listdir(ckpt_dir)),
                       reverse=True,
                       key=lambda x: int(x[5:].split('.')[0]))[:num_checkpoints]:
        srcfile = os.path.join(ckpt_dir, file)
        tempfile = os.path.join(target_dir, "_temp")
        destfile = os.path.join(target_dir, file)
        shutil.copy(srcfile, tempfile)
        os.rename(tempfile, destfile)
