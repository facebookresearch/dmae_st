# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import torch
import math
import numpy as np
from typing import Callable

class CustomLRScheduler():
    def __init__(self,
                 fn: Callable,
                 optimizer: torch.optim.Optimizer):
        self.fn = fn
        self.optimizer = optimizer
    
    def __call__(self, step: int):
        lr = self.fn(step)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def lr_fn_warmup(base_lr: float,
                 train_steps: int,
                 warmup_steps: int,
                 decay_type: str="cosine",
                 min_lr: float=1e-5) -> float:
    """
    Learning rate with warmup and cosine/linear options.

    Args:
        base_lr (float): base learning rate.
        train_steps (int): total training steps
        warmup_steps (int): number of warmup steps
        min_lr (float, optional): [description]. Defaults to 1e-5.

    Returns:
        Callable: Learning rate function.
    """

    def step_fn(step: int):
        lr = base_lr

        progress = (step - warmup_steps) / float(train_steps - warmup_steps)
        progress = np.clip(progress, 0., 1.)

        if decay_type == 'linear':
            lr = min_lr + (lr-min_lr) * (1. - progress)
        elif decay_type == 'cosine': 
            lr = min_lr + (lr-min_lr) * 0.5 * (1. + math.cos(math.pi * progress))

        if warmup_steps:
            lr = lr * min(1., step / warmup_steps)

        return lr
    
    return step_fn

def lr_fn_warmup_stair(base_lr: float,
                       warmup_steps: int,
                       decay_steps: int,
                       decay_rate: float) -> float:
    """
    Learning rate with warmup and staircase exponential decay.

    Args:
        base_lr (float): base learning rate
        warmup_steps (int): number of warmup steps.
        decay_steps (int): frequency of exponential decay
        decay_rate (float): rate of exponential decay applied at decay_steps.
    """
    def step_fn(step: int):
        lr = base_lr
        
        eff_steps = step - warmup_steps

        lr = lr * (decay_rate ** max(0., math.floor(eff_steps / decay_steps)))

        if warmup_steps:
            lr = lr * min(1., step / warmup_steps)
        
        return lr

    return step_fn

def lr_fn_stair(base_lr: float,
                decay_steps: int,
                decay_rate: float) -> float:
    """
    Learning rate with warmup and staircase exponential decay.

    Args:
        base_lr (float): base learning rate
        warmup_steps (int): number of warmup steps.
        decay_steps (int): frequency of exponential decay
        decay_rate (float): rate of exponential decay applied at decay_steps.
    """
    def step_fn(step: int):
        lr = base_lr
        lr = lr * (decay_rate ** max(0., math.floor(step / decay_steps)))

        return lr

    return step_fn

def lr_fn_cos(base_lr: float,
              total_steps: int) -> float:
    """
    Learning rate with cosine schedule.

    Args:
        base_lr (float): base learning rate
        total_steps (int): totals teps

    Returns:
        float: [description]
    """
    def step_fn(step: int):
        lr = base_lr
        lr = lr * 0.5 * (1. + math.cos(math.pi * step / total_steps))
        
        return lr
    return step_fn

def lr_fn_warmup_cos(base_lr: float,
              warmup_steps: int,
              total_steps: int) -> float:
    """
    Learning rate cosine schedule and warmup.

    Args:
        base_lr (float): learning rate, inclusive of warmup
        warmup_steps (int): warmup steps
        total_steps (int): total steps

    Returns:
        float: [description]
    """
    def step_fn(step: int):
        lr = base_lr
        
        eff_step = max(0, step - warmup_steps)
        
        lr = lr * 0.5 * (1. + math.cos(math.pi * eff_step / total_steps))
        
        if warmup_steps:
            lr = lr * min(1., step / warmup_steps)

        return lr
    return step_fn
