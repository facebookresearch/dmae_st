# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from iopath.common.file_io import g_pathmgr as pathmgr
from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
# from tensorboard.fb.manifoldio import ManifoldFileSystem
from torch.utils.tensorboard import SummaryWriter
from dmae.util.kinetics_bbox import KineticsBbox
from dmae.util.kinetics_v2 import KineticsV2

from mae.util import video_vit
from mae.util.kinetics import Kinetics

assert timm.__version__ == "0.3.2"  # version check
import dmae.models_dmae as models_dmae
import mae.util.logging as logging
import mae.util.misc as misc
from dmae.engine_pretrain import train_one_epoch
from mae.util.misc import NativeScalerWithGradNormCount as NativeScaler, get_mask_ratio
from dmae.main_pretrain import get_args_parser

def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True

    dataset_train = KineticsBbox(
        mode="pretrain",
        path_to_data_dir=args.data_dir_kinetics,
        path_to_bbox_dir=args.data_dir_bbox,
        num_retries=args.num_retries,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        crop_size=args.input_size,
        repeat_aug=args.repeat_aug,
        pre_crop_size=256, #ugly parameter set, but minimizes torchvision-based loading errors
        random_horizontal_flip=args.horizontal_flip,
        jitter_aspect_relative=args.jitter_aspect_relative,
        jitter_scale_relative=args.jitter_scales_relative,
        backend=args.backend # torchvision -- pyav is considerably slower.
    )
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if misc.is_main_process():
        if args.log_dir is None:
            args.log_dir = os.path.join(args.output_dir, "logs")
        try:
            pathmgr.mkdirs(args.log_dir)
        except Exception as e:
            pass

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model_fn = models_dmae.__dict__[args.model]
    model = model_fn(**vars(args),)

    model.to(device)

    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = run_one_epoch(
            model,
            data_loader_train,
            device,
            epoch,
            args=args,
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())

def run_one_epoch(
    model,
    data_loader,
    device,
    epoch,
    args
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter=" ")
    
    header = "".format(epoch)
    print_freq = 20

    for data_iter_step, (samples, bbox_map, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        bbox_map = bbox_map.to(device, non_blocking=True)
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)
            if len(bbox_map.shape) == 6:
                bbox_map = bbox_map.reshape(b * r, 1, t, h, w)

        mask_ratio = get_mask_ratio(args, epoch)
        meters = {}
        temperatures = {"125": 1.25, "100": 1.00, "075": 0.75, "050": 0.5}
        for key in temperatures:
            temperature = temperatures[key]
            p_in, p_mask_in, p_out, p_mask_out = weighted_masking(
                model,
                samples,
                bbox_map,
                temperature,
                mask_ratio
            )
            meters.update({
                f"in": p_in,
                f"{key}in_mask": p_mask_in,
                f"out": p_out,
                f"{key}out_mask": p_mask_out
            })
        
        torch.cuda.synchronize()
        for key in sorted(meters.keys()):
            metric_logger.update(**{key:meters[key]})
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def weighted_masking(
    model: models_dmae.DirectedMaskedAutoencoderViT,
    x,
    binary_map,
    temperature,
    mask_ratio,
):
    x = model.patch_embed(x)
    N, T, L, C = x.shape
    x = x.reshape(N, T * L, C)
    
    binary_map = binary_map + 1.0 # Add 1 so that low-pri regions aren't 0. Rely on temperature to modulate
    patch_binary_map = model.patch_binary_map(binary_map) # [B, T, H*W, 1]
    patch_weights = torch.softmax(patch_binary_map[..., 0] / temperature,
                                  dim=-1)[..., None].reshape(N, T * L, 1)
    
    _, mask, _, _ = model.weighted_masking(x, patch_weights, mask_ratio)
    
    patch_binary_map = patch_binary_map.view(N, -1)
    patches_in, patches_out = patch_binary_map >= 1.5, patch_binary_map < 1.5
    
    mask = mask.type(torch.bool)
    
    num_patches = model.patch_embed.num_patches
    n_patches_in = patches_in.type(torch.float32).sum(-1)
    n_patches_out = patches_out.type(torch.float32).sum(-1)
    
    n_mask_in = torch.logical_and(mask, patches_in).type(torch.float32).sum(-1)
    n_mask_out = torch.logical_and(mask, patches_out).type(torch.float32).sum(-1)
    
    return (
        (n_patches_in / num_patches).mean(),
        (n_mask_in / n_patches_in).nanmean(),
        (n_patches_out / num_patches).mean(),
        (n_mask_out / n_patches_out).nanmean()
    )