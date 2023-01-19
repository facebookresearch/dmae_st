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
from typing import Iterable

import torch
from dmae.util.utils import compute_masking_ratios

import mae.util.misc as misc
import mae.util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    print_freq = 20
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter("mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbx", misc.SmoothedValue(window_size=print_freq, fmt="{avg:.2f}({global_avg:.2f})"))
    metric_logger.add_meter("outbx", misc.SmoothedValue(window_size=print_freq, fmt="{avg:.2f}({global_avg:.2f})"))
    metric_logger.add_meter("temp", misc.SmoothedValue(window_size=print_freq, fmt="{avg:.2f}({global_avg:.2f})"))
    metric_logger.add_meter("inmsk", misc.SmoothedValue(window_size=print_freq, fmt="{avg:.2f}({global_avg:.2f})"))
    metric_logger.add_meter("outmsk", misc.SmoothedValue(window_size=print_freq, fmt="{avg:.2f}({global_avg:.2f})"))
    metric_logger.add_meter("empty", misc.SmoothedValue(window_size=1, fmt="{total:.0f}"))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, bbox_map, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        bbox_map = bbox_map.to(device, non_blocking=True)

        mask_ratio = args.mask_ratio
        temperature = args.temperature
        
        with torch.cuda.amp.autocast(enabled=not args.fp32):
            loss, _, mask = model(samples,
                                  bbox_map,
                                  mask_ratio=args.mask_ratio,
                                  temperature=temperature)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        p_in, p_in_mask, p_out, p_out_mask, n_empty = compute_masking_ratios(
            mask,
            model.module.patch_binary_map(bbox_map),
            model.module.patch_embed.num_patches
        )

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(mask_ratio=mask_ratio)
        metric_logger.update(inbx=p_in)
        metric_logger.update(outbx=p_out)
        metric_logger.update(temp=temperature)
        metric_logger.update(inmsk=p_in_mask if torch.isfinite(p_in_mask).all() else 0.) # do not store NaNs
        metric_logger.update(outmsk=p_out_mask)
        metric_logger.update(empty=n_empty)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('TRAIN_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('TRAIN_lr', lr, epoch_1000x)
            log_writer.add_scalar("TRAIN_mask_ratio", mask_ratio, epoch_1000x)
            log_writer.add_scalar("TRAIN_in_box", p_in, epoch_1000x)
            log_writer.add_scalar("TRAIN_out_box", p_out, epoch_1000x)
            log_writer.add_scalar("TRAIN_temperature", temperature, epoch_1000x)
            log_writer.add_scalar("TRAIN_in_mask", p_in_mask, epoch_1000x)
            log_writer.add_scalar("TRAIN_out_mask", p_out_mask, epoch_1000x)
            log_writer.add_scalar("TRAIN_empty", n_empty, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}