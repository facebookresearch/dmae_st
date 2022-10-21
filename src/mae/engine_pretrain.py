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
from iopath.common.file_io import g_pathmgr as pathmgr

import mae.util.logging as logging
import mae.util.lr_sched as lr_sched
import mae.util.misc as misc
from mae.util.misc import get_mask_ratio, plot_input


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    visualize=False,
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
    metric_logger.add_meter(
        "mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)

        visualize_freq = 50
        mask_ratio = get_mask_ratio(args, epoch)
        with torch.cuda.amp.autocast(enabled=not fp32):
            loss, _, _, vis = model(
                samples,
                mask_ratio=mask_ratio,
                # visualize=visualize and data_iter_step % visualize_freq == 0,
                visualize=False,
            )

        if visualize and data_iter_step % visualize_freq == 0:
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                if not pathmgr.exists(f"{args.output_dir}/vis"):
                    try:
                        pathmgr.mkdirs(f"{args.output_dir}/vis")
                    except Exception as e:
                        pass
                vis = vis.detach().cpu().permute(0, 1, 3, 2, 4, 5)
                for i in range(vis.shape[0]):
                    # B 3 C T H W -> B 3 T C H W
                    plot_input(
                        vis[i],
                        path=f"{args.output_dir}/vis/{epoch}_{data_iter_step}_{i}.jpg",
                    )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            for _ in range(args.num_checkpoint_del):
                try:
                    path = misc.get_last_checkpoint(args)
                    pathmgr.rm(path)
                    print(f"remove checkpoint {path}")
                except Exception as e:
                    pass
            raise Exception("Loss is {}, stopping training".format(loss_value))

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        # metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        # metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        # metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(mask_ratio=mask_ratio)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000 * args.repeat_aug
            )
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
