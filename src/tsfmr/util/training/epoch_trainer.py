# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from time import time
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util.training.lr_util import CustomLRScheduler
from util.training.meters import AverageMeter


class EpochTrainer:
    def __init__(
        self,
        # training
        train_fn: Callable,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        epochs: int,
        print_freq: int,
        # checkpointing
        ckpt_fn: Callable,
        # summary saving
        summary_writer: SummaryWriter
    ):
        """
        Trainer class.
        Initialize with minimum information for training a job. Must include checkpointing.
        Other functions can be added through self.regiser_{} methods.

        Args:
            train_step (Callable): Performs a single step of gradient descent, given step.
            model (torch.nn.Module): model.
            optimizer (torch.optim.Optimizer): optimizer.
            dataloader (object): training dataloader
            epochs (int): number of training steps.
            print_freq (int): frequency of printing.
            ckpt_fn (Callable): Saves checkpoints given step.
            ckpt_freq (int): frequency of checkpoint saving.
            summary_writer (object): Summary writer.
        """
        
        self.train_fn = train_fn
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.epochs = epochs
        self.print_freq = print_freq

        self.ckpt_fn = ckpt_fn

        self.summary_writer = summary_writer

        self.save = False
        self.eval = False
        self.lr_scheduler = None
        self.distributed = False # If not distributed, rank = 0

        print("Trainer Initialized.")

    def reg_save(
        self,
        save_fn: Callable,
        save_freq: int,
    ):
        """
        Register save function.

        Args:
            save_fn (Callable): Arguments (step, data_batch, model_output, step_output).
            save_freq (int): Frequency to call save_fn.
        """
        self.save = True
        self.save_fn = save_fn
        self.save_freq = save_freq

        print("Trainer: Save functionality registered.")

    def reg_eval(
        self,
        eval_fn: Callable,
        eval_freq: int=1,
    ):
        """
        Register eval function.

        Args:
            eval_fn (Callable): Arguments (step).
            eval_freq (int): frequency, defaults t o1
        """
        self.eval = True
        self.eval_fn = eval_fn
        self.eval_freq = eval_freq

        print("Trainer: Eval functionality registered.")

    def reg_lr_scheduler(
        self, 
        lr_scheduler: CustomLRScheduler
    ):
        """
        Register custom learning rate scheduler.

        Args:
            lr_scheduler (CustomLRScheduler): Learning rate scheduler.
        """
        self.lr_scheduler = lr_scheduler

    def reg_rank(
        self,
        rank: int
    ):
        """
        Register rank for process in distributed training.

        Args:
            rank (int): Rank, int >= 0
        """
        self.distributed = True
        self.rank = rank

    def train(self, start_epoch):
        print("Trainer: begin training.")

        output_meter_dict = {}
        meter_data_time = AverageMeter(name="sys_DataTime")
        meter_model_time = AverageMeter(name='sys_ModelTime')
        meter_step_time = AverageMeter(name="sys_StepTime")

        summary_writer = self.summary_writer

        data_init, model_init = False, False
        steps_per_epoch = len(self.dataloader)
        for epoch in range(start_epoch, self.epochs):
            data_iterator = self.dataloader.__iter__()
            print("Trainer: starting epoch {}".format(epoch))
            
            for step in range(steps_per_epoch):
                start_time = time()
                
                batch = next(data_iterator)
                data_time = time() - start_time
                if not data_init:
                    print("Trainer: dataset initialized, first batch acquired in: (Time: {0:4.3f})".format(data_time))
                data_init = True
            
                # training step
                model_output, step_output = self.train_fn(step, batch)
                model_time = time() - start_time - data_time
                step_time = time() - start_time 
                if not model_init:
                    print("Trainer: model initialized, first train step completed in: (Time: {0:4.3f})".format(model_time))
                    model_init = True
                
                # Update learning rate
                if self.lr_scheduler:
                    self.lr_scheduler(epoch * steps_per_epoch + step)
                
                # Check numerics
                try:
                    for value in step_output.values():
                        assert torch.all(torch.isfinite(value))
                except Exception as e:
                    error_dict = {}
                    for key, value in step_output.items():
                        error_dict[key] = torch.all(torch.isfinite(value))
                    print("Step[{}] -- Error dict: {}".format(step, error_dict))
                    raise e
                
                # Update meters
                for key, value in step_output.items():
                    if key not in output_meter_dict:
                        output_meter_dict[key] = AverageMeter(name=key)
                    output_meter_dict[key](value)
                meter_data_time(data_time)
                meter_model_time(model_time)
                meter_step_time(step_time)

                # Print + Summary writing on self.print_freq
                if step % self.print_freq == 0:
                    curr_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Printing
                    print("Epochs:  [{epoch:3d}/{total_epochs:3d}]  Step: [{step:6d}/{total_steps}]  |  [Time]  Step {steptime:5.3f}  Data {datatime:5.3f}  Model {modeltime:4.3f}  |  {metrics}  |  LR {lr:5.3f}".format(
                        epoch=epoch,
                        total_epochs=self.epochs,
                        step=step,
                        total_steps=steps_per_epoch,
                        datatime=meter_data_time.result(),
                        modeltime=meter_model_time.result(),
                        steptime=meter_step_time.result(),
                        metrics="  ".join(["{} {:7.3f}".format(k, v.result()) for k, v in output_meter_dict.items()]),
                        lr=curr_lr
                    ))

                    if not self.distributed or self.rank == 0:
                        # Summary Writing
                        summary_writer.add_scalar(meter_data_time.name, meter_data_time.result(), global_step=epoch*steps_per_epoch + step)
                        summary_writer.add_scalar(meter_model_time.name, meter_model_time.result(), global_step=epoch*steps_per_epoch + step)
                        summary_writer.add_scalar(meter_step_time.name, meter_step_time.result(), global_step=epoch*steps_per_epoch + step)
                        summary_writer.add_scalar("LearningRate", curr_lr, global_step=epoch*steps_per_epoch + step)
                        summary_writer.add_scalar("Epoch", epoch, global_step=epoch*steps_per_epoch + step)

                        for meter_name, meter in output_meter_dict.items():
                            summary_writer.add_scalar(meter_name, meter.result(), global_step=epoch*steps_per_epoch + step)
                            
                        if self.save and step % self.save_freq == 0:
                            self.save_fn(epoch*steps_per_epoch + step, batch, model_output, step_output)
                    
                    for meter in output_meter_dict.values():
                        meter.reset()
                    meter_data_time.reset()
                    meter_model_time.reset()
                    meter_step_time.reset()
            
            if not self.distributed or self.rank == 0:
                self.ckpt_fn(epoch)
                
                if self.eval and (epoch % self.eval_freq == 0 or epoch == self.epochs-1):
                    eval_start_time = time()
                    eval_outputs = self.eval_fn(epoch)
                    eval_time = time() - eval_start_time

                    print("#EVAL  |  [EvalTime] {eval_time:4.3f}  |  {metrics}".format(
                        eval_time=eval_time,
                        metrics="  ".join(["{} {:7.3f}".format(k, v) for k, v in eval_outputs.items()])
                    ))

                    for eval_output_name, eval_output in eval_outputs.items():
                        summary_writer.add_scalar("EVAL_"+eval_output_name, eval_output, global_step=(epoch+1)*steps_per_epoch)
                
            

                
                
            