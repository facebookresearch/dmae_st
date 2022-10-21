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
from mae.util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--accum_iter", default=1, type=int, 
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",)

    # Model parameters
    parser.add_argument("--model", default="dmae_vit_large_patch16", type=str, metavar="MODEL",
                        help="Name of model to train")
    parser.add_argument("--input_size", default=224, type=int, 
                        help="images input size")
    parser.add_argument("--mask_schedule", default="const", type=str,
                        help="const or cos",)
    parser.add_argument("--mask_ratio", default=0.90, type=float,
                        help="Masking ratio (percentage of removed patches).")
    parser.add_argument("--mask_ratio_end", default=0.90, type=float,
                        help="UNUSED: Masking ratio (percentage of removed patches) final value for cos schedule.",)
    parser.add_argument("--norm_pix_loss", action="store_true", 
                        help="Use (per-patch) normalized pixels as targets for computing loss",)
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, 
                        help="weight decay (default: 0.05)")
    parser.add_argument("--lr", type=float, default=None, metavar="LR",
                        help="learning rate (absolute lr)",)
    parser.add_argument("--blr",type=float, default=4e-4, metavar="LR",
                        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",)
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR",
                        help="lower lr bound for cyclic schedulers that hit 0",)
    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N",
                        help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--data_dir_kinetics", default="/datasets01/kinetics/092121/400", type=str,
                        help="Kinetics dataset path",)
    parser.add_argument("--data_dir_bbox", default="/data/home/alexnw/alexnw/projects/skel_act_recg/data/kinetics_400_annotations-videos", type=str,
                        help="bounding boxes path")

    parser.add_argument("--output_dir", default="./output_dir", 
                        help="path where to save, empty for no saving",)
    parser.add_argument("--log_dir", default=None,
                        help="path where to tensorboard log",)
    parser.add_argument("--device", default="cuda", 
                        help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="start epoch")
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--fb_env", action="store_true")

    # Tunable Configs
    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=4, type=int)
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--t_patch_size", default=2, type=int,
                        help="""Temporal kernel size when projecting into patches""")
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=5, type=int)
    parser.add_argument("--sampling_rate", default=4, type=int)
    # parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--repeat_aug", default=4, type=int)

    # Video related configs
    parser.add_argument("--dist_url", default="env://", 
                        help="url used to set up distributed training")

    # Options include: s, st, tube
    parser.add_argument("--mask_type", default="directed",type=str,
                        help="video masking strategy, s, st, tube")

    # Dataset parameters
    parser.add_argument("--encoder_attn", default="AttentionWithCls",type=str,
                        help="AttentionMode for the Encoder")
    parser.add_argument("--decoder_attn", default="AttentionOrg",type=str,
                        help="AttentionMode for the Decoder")
    parser.add_argument("--clip_grad", type=float, default=0.02,
                        help="Gradient clipping to value.")
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=0, type=int)
    parser.add_argument("--learnable_pos_embed", action="store_true")
    parser.set_defaults(learnable_pos_embed=True)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument("--trunc_init", action="store_true",)
    parser.add_argument("--fp32", action="store_true",)
    parser.set_defaults(fp32=True)
    parser.add_argument("--jitter_scales_relative", default=[0.5, 1.0], type=float, nargs="+",
                        help="Spatial resize and crop min and max scale.")
    parser.add_argument("--jitter_aspect_relative", default=[0.75, 1.3333], type=float, nargs="+",
                        help="Spatial resize min and max aspect ratio.")
    parser.add_argument("--horizontal_flip", default=True, action="store_true",
                        help="Horizontal flipping.")
    parser.add_argument("--beta", default=[0.9, 0.95], type=float, nargs="+",
                        help="AdamW beta values")
    parser.add_argument("--pred_t_dim", type=int, default=8,
                        help="temporal dimension of decoder output patch")
    parser.add_argument("--cls_embed", action="store_true")
    parser.add_argument("--no_cls_embed", action="store_false", dest="cls_embed")
    parser.set_defaults(cls_embed=True)
    
    parser.add_argument("--temperature_schedule", default="const", type=str,
                        help="const or cos")
    parser.add_argument("--temperature", default=1.0, type=float,
                        help="temperature to tune sharpness of directed masking skew")
    parser.add_argument("--temperature_end", default=1.0, type=float,
                        help="Temperature end when schedule != const")
    parser.add_argument("--flip_mask", default=False, action="store_true",
                        help="Flip the bounding box binary mask.")
    
    parser.add_argument("--num_retries", type=int, default=12,
                        help="Number of times to retry finding and loading a video example.")
    parser.add_argument("--backend", default="torchvision", type=str,
                        help="Video loading backend. (torchvision, pyav)")
    
    return parser


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
        # register_filesystem("manifold", ManifoldFileSystem())
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

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

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters: {}".format(total_params))

    eff_batch_size = (
        args.batch_size * args.accum_iter * misc.get_world_size() * args.repeat_aug
    )

    assert (args.lr is not None) != (args.blr is not None) # only one is specified
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            # find_unused_parameters=True,
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        args.weight_decay,
        bias_wd=args.bias_wd,
    )
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    beta = args.beta
    optimizer = torch.optim._multi_tensor.AdamW(
        param_groups,
        lr=args.lr,
        betas=beta,
    )
    print(optimizer)
    loss_scaler = NativeScaler(fp32=args.fp32)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            visualize=False,
            fp32=args.fp32,
        )
        if args.output_dir and (
            epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs
        ):
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def launch_one_thread(
    local_rank,
    shard_rank,
    num_gpus_per_node,
    num_shards,
    init_method,
    output_path,
    opts,
    stats_queue,
):
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print(opts)
    args = get_args_parser()
    args = args.parse_args(opts)
    args.rank = shard_rank * num_gpus_per_node + local_rank
    args.world_size = num_shards * num_gpus_per_node
    args.gpu = local_rank
    args.dist_url = init_method
    args.output_dir = output_path
    output = main(args)
    stats_queue.put(output)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
