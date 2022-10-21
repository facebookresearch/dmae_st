# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
from iopath.common.file_io import g_pathmgr as pathmgr
from timm.data import Mixup
from timm.utils import accuracy

import mae.util.logging as logging
import mae.util.lr_sched as lr_sched
import mae.util.misc as misc
from mae.util.logging import master_print as print


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
    fp32=False,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter(
    #     "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    # )
    # metric_logger.add_meter(
    #     "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    # )
    # metric_logger.add_meter(
    #     "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    # )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, bbox_map, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.view(b * r, c, t, h, w)
            targets = targets.view(b * r)
            if len(bbox_map.shape) == 6:
                bbox_map = bbox_map.reshape(b * r, 1, t, h, w)
                
        if args.cpu_mix:
            raise NotImplementedError("augmentations not available for bboxes")
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        else:
            samples = samples.to(device, non_blocking=True)
            bbox_map = bbox_map.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                raise NotImplementedError("mixup not available for bboxes")
                samples, targets = mixup_fn(samples, targets)

        # misc.plot_input(samples.cpu().permute(0, 2, 1, 3, 4), path=f"vis/{data_iter_step}.jpg")

        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples, bbox_map)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        # metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        # metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        # metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000 * args.repeat_aug
            )
            log_writer.add_scalar("FINETUNE_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("FINETUNE_lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        bbox_map = batch[1]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        bbox_map = bbox_map.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if len(images.shape) == 6:
            b, r, c, t, h, w = images.shape
            images = images.view(b * r, c, t, h, w)
            target = target.view(b * r)
            if len(bbox_map.shape) == 6:
                bbox_map = bbox_map.reshape(b * r, 1, t, h, w)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, bbox_map)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
