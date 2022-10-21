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
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data as torchdata
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from iopath.common.file_io import g_pathmgr as pathmgr
from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
# from tensorboard.fb.manifoldio import ManifoldFileSystem
from torch.utils.tensorboard import SummaryWriter
from dmae.util.decoder.transform import horizontal_flip
from dmae.util.kinetics_bbox import KineticsBbox

from mae.util import video_vit
from mae.util.kinetics import Kinetics
from mae.util.lars import LARS
from mae.util.logging import master_print as print

assert timm.__version__ == "0.3.2"  # version check
# from pytorchvideo.transforms.mix import MixVideo
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_

import dmae.models_masked_vit as models_vit
import mae.util.lr_decay as lrd
import mae.util.misc as misc
from dmae.engine_masked_finetune import evaluate, train_one_epoch
# from util.datasets import build_dataset
from mae.util.decoder.mixup import MixUp as MixVideo
from mae.util.misc import NativeScalerWithGradNormCount as NativeScaler
from mae.util.pos_embed import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser("MAE fine-tuning for image classification", add_help=False)
    
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--accum_iter", default=1, type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",)

    # Model parameters
    parser.add_argument("--model",default="masked_vit_large_patch16", type=str, metavar="MODEL",
                        help="Name of model to train",)

    parser.add_argument("--input_size", default=224, type=int,
                        help="images input size")

    parser.add_argument("--dropout", type=float, default=0.,)

    parser.add_argument("--drop_path_rate", type=float, default=0., metavar="PCT",
                        help="Drop path rate (default: 0.1)",)

    # Optimizer parameters
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM",
                        help="Clip gradient norm (default: None, no clipping)",)
    parser.add_argument("--weight_decay", type=float, default=0., 
                        help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR",
                        help="learning rate (absolute lr)",)
    parser.add_argument("--blr", type=float, default=1e-1, metavar="LR",
                        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",)
    parser.add_argument("--layer_decay", type=float, default=0., 
                        help="layer-wise lr decay from ELECTRA/BEiT",)

    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="LR",
                        help="lower lr bound for cyclic schedulers that hit 0",)

    parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N",
                        help="epochs to warmup LR")

    # Augmentation parameters
    parser.add_argument( "--color_jitter", type=float, default=None, metavar="PCT", 
                        help="Color jitter factor (enabled only when not using Auto/RandAug)", )
    parser.add_argument( "--aa", type=str, default="", metavar="NAME", 
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)', ),
    parser.add_argument( "--smoothing", type=float, default=0.1, 
                        help="Label smoothing (default: 0.1)" )

    # * Random Erase params
    parser.add_argument( "--reprob", type=float, default=0., metavar="PCT",
                        help="Random erase prob (default: 0.25)", )
    parser.add_argument( "--remode", type=str, default="pixel", 
                        help='Random erase mode (default: "pixel")', )
    parser.add_argument( "--recount", type=int, default=1, 
                        help="Random erase count (default: 1)" )
    parser.add_argument( "--resplit", action="store_true", default=False, 
                        help="Do not random erase first (clean) augmentation split", )

    # * Mixup params
    parser.add_argument( "--mixup", type=float, default=0, 
                        help="mixup alpha, mixup enabled if > 0." )
    parser.add_argument( "--cutmix", type=float, default=0,
                        help="cutmix alpha, cutmix enabled if > 0." )
    parser.add_argument( "--cutmix_minmax", type=float, nargs="+", default=None,
                        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)", )
    parser.add_argument( "--mixup_prob", type=float, default=1.0, 
                        help="Probability of performing mixup or cutmix when either/both is enabled", )
    parser.add_argument( "--mixup_switch_prob", type=float, default=0.5, 
                        help="Probability of switching to cutmix when both mixup and cutmix enabled", )
    parser.add_argument( "--mixup_mode", type=str, default="batch", 
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"', )

    # * Finetuning params
    parser.add_argument("--finetune", default="", 
                        help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument( "--cls_token", action="store_false", dest="global_pool", 
                        help="Use class token instead of global pool for classification", )

    # Dataset parameters
    parser.add_argument("--data_dir_kinetics", default="/datasets01/kinetics/092121/400", type=str,
                        help="Kinetics dataset path",)
    parser.add_argument("--data_dir_bbox", default="/data/home/alexnw/alexnw/projects/skel_act_recg/data/kinetics_400_annotations-videos", type=str,
                        help="bounding boxes path")
    parser.add_argument( "--num_classes", default=400, type=int, 
                        help="number of the classification types", )

    parser.add_argument( "--output_dir", default="./output_dir", 
                        help="path where to save, empty for no saving", )
    parser.add_argument( "--log_dir", default=None, 
                        help="path where to tensorboard log", )
    parser.add_argument( "--device", default="cuda", 
                        help="device to use for training / testing" )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="",
                        help="resume from checkpoint")

    parser.add_argument( "--start_epoch", default=0, type=int, metavar="N", 
                        help="start epoch" )
    parser.add_argument("--eval", action="store_true", 
                        help="Perform evaluation only")
    parser.add_argument( "--dist_eval", action="store_true", default=False, 
                        help="Enabling distributed evaluation (recommended during training for faster monitor", )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument( "--pin_mem", action="store_true", 
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.", )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument( "--world_size", default=1, type=int, 
                        help="number of distributed processes" )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", 
                        help="url used to set up distributed training")

    # Video related configs
    parser.add_argument("--fb_env", action="store_true")
    parser.add_argument("--rand_aug", default=False, action="store_true")
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=2, type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--repeat_aug", default=1, type=int)
    parser.add_argument("--encoder_attn", default="AttentionWithCls", type=str)
    parser.add_argument("--cpu_mix", action="store_true")
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument( "--rel_pos_init_std", type=float, default=0.02,)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.add_argument( "--fp32", action="store_true")
    parser.add_argument( "--jitter_scales_relative", default=[0.08, 1.0], type=float, nargs="+", ) 
    parser.add_argument( "--jitter_aspect_relative", default=[0.75, 1.3333], type=float, nargs="+", )
    parser.add_argument("--horizontal_flip", default=True, action="store_true",
                        help="Horizontal flipping.")
    parser.add_argument("--cls_embed", action="store_true")
    
    # Finetuning dataset subsampling
    parser.add_argument("--finetune_fraction", default=1.0, type=float,
                        help="[0, 1.] for the amount of the full training set to fine tune on.")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="location for checkpoint paths")
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
    
    # set torchvision backend to ffmpeg, video_reader
    torchvision.set_video_backend("video_reader")

    cudnn.benchmark = True

    dataset_train = KineticsBbox(
        mode="pretrain",
        path_to_data_dir=args.data_dir_kinetics,
        path_to_bbox_dir=args.data_dir_bbox,
        num_retries=12,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        crop_size=args.input_size,
        repeat_aug=args.repeat_aug,
        pre_crop_size=256, #ugly parameter set, but minimizes torchvision-based loading errors
        random_horizontal_flip=args.horizontal_flip,
        jitter_aspect_relative=args.jitter_aspect_relative,
        jitter_scale_relative=args.jitter_scales_relative,
        backend="torchvision" # torchvision -- pyav is considerably slower.
    )
    
    dataset_amt = int(len(dataset_train) * args.finetune_fraction)
    dataset_train, _ = torchdata.random_split(dataset_train,
                                              [dataset_amt,
                                               len(dataset_train) - dataset_amt],
                                              generator=torch.Generator().manual_seed(seed))
    
    dataset_val = KineticsBbox(
        mode="val",
        path_to_data_dir=args.data_dir_kinetics,
        path_to_bbox_dir=args.data_dir_bbox,
        num_retries=12,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        crop_size=args.input_size,
        pre_crop_size=256,
        random_horizontal_flip=args.horizontal_flip,
        jitter_aspect_relative=args.jitter_aspect_relative,
        jitter_scale_relative=args.jitter_scales_relative,
        backend="torchvision"
    )

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and not args.eval:
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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = MixVideo(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            mix_prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )

    model = models_vit.__dict__[args.model](
        **vars(args),
    )

    if misc.get_last_checkpoint(args) is None and args.finetune and not args.eval:
        with pathmgr.open(args.finetune, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if "model" in checkpoint.keys():
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint["model_state"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # checkpoint_model = misc.inflate(checkpoint_model, state_dict)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = (
        args.batch_size * args.accum_iter * misc.get_world_size() * args.repeat_aug
    )

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    for _, p in model_without_ddp.named_parameters():
        p.requires_grad = False
    for _, p in model_without_ddp.head.named_parameters():
        p.requires_grad = True

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()]
        )
        model_without_ddp = model.module
        
    n_trainable = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print("number of trainable params: %.2f" % (n_trainable))

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = model_without_ddp.head.parameters()
    
    # optimizer = LARS(param_groups, lr=args.lr)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr)
    loss_scaler = NativeScaler(fp32=args.fp32)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            args=args,
            fp32=args.fp32,
        )
        if args.output_dir:
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
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
    # clean up mem
    gc.collect()
    torch.cuda.empty_cache()

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
