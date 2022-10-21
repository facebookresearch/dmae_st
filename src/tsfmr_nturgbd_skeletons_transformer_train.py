import argparse
from argparse import Namespace
from functools import partial
import os
from pprint import pprint
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

from tsfmr.configs.configs import get_config, merge_args_into_config
from tsfmr.data.nturgbd_skeletons import NTURGBDSkeletons
from tsfmr.data.transforms.uniform_sampling import UniformSampling
from tsfmr.model.classification_transformer import ClassificationTransformer
from tsfmr.util.training.checkpointing import checkpoint_and_load
from tsfmr.util.training.distributed import set_distributed_mode
from tsfmr.util.training.epoch_trainer import EpochTrainer
from tsfmr.util.training.experiment import experiment_setup
from tsfmr.util.training.lr_util import CustomLRScheduler, lr_fn_warmup_cos
from tsfmr.util.training.trainer import Trainer

parser = argparse.ArgumentParser()

# Required Experiment Args
parser.add_argument("--dir", required=True, type=str,
                    help="Base experiment directory where outputs are placed.")
parser.add_argument("--data_dir", type=str, default="data/nturgb+d_skeletons",
                    help="""Path to dataset.""")
parser.add_argument("--config", type=str, default="default",
                    help="""Config string, delimited by dots.""")

# Distributed Args
parser.add_argument("--dist", action="store_true",
                    help="""Set to true if run in distributed mode.""")
parser.add_argument("--dist_url", type=str,
                    help="""URL used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html.""")
parser.add_argument("--world_size", type=int, 
                    help="""Number of processes: it is set automatically and should not be passed as argument.""")
parser.add_argument("--rank", type=int, 
                    help="""Rank of this process: it is set automatically and should not be passed as argument.""")
parser.add_argument("--port", type=int,
                    help="""Optional argument to specify the port for dist_url; useful to prevent collisions.""")

# Logging Args
parser.add_argument("--print_freq", default=50, type=int,
                    help="""Number of steps per logging action during training.""")
parser.add_argument("--save_freq", default=1000, type=int,
                    help="""Number of steps per save action during training.""")
parser.add_argument("--eval_freq", default=10, type=int,
                    help="""Number of epochs per evaluation step.""")

### Below here are experiment-specific configs. ###

# Dataset Args
parser.add_argument("--eval_type", type=str,
                    help="""Evaluation type: (xsub, xview).""")
parser.add_argument("--num_frames", type=int,
                    help="""Number of frames to downsample to.""")

# Dataloader args
parser.add_argument("--batch_size", type=int,
                    help="""Batch size, per node for data loading.""")

# Optimization Args
parser.add_argument("--epochs", type=int,
                    help="""Total number of training steps.""")
parser.add_argument("--base_lr", type=float,
                    help="Base learning rate.")
parser.add_argument("--lr_decay_min", type=float,
                    help="Minimum learning rate to decay to.")
parser.add_argument("--lr_warmup_epochs", type=int,
                    help="Number of steps to warmup to base learning rate.")

# Model Args
parser.add_argument("--embed_dim", type=int,
                    help=""""Transformer embedding dimension.""")
parser.add_argument("--depth", type=int,
                    help="""Transformer layer depth.""")
parser.add_argument("--num_heads", type=int,
                    help="""Number of heads in MHA.""")
parser.add_argument("--mlp_ratio", type=int,
                    help="""MLP expansion ratio in transformer block.""")
parser.add_argument("--qkv_bias", type=bool,
                    help="""Include bias in QKV attention.""")
parser.add_argument("--drop_rate", type=float,
                    help="""Dropout rate for dropout layers.""")
parser.add_argument("--attn_drop_rate", type=float,
                    help="""Dropout rate for attention QKV attention matrix.""")
parser.add_argument("--drop_path_rate", type=float,
                    help="""Stochastic depth by temporal dropout.""")
parser.add_argument("--norm_layer", type=str,
                    help="""Norm layer used (layer, batch, group).""")
parser.add_argument("--act_layer", type=str,
                    help="""Activation layer (ReLU, GELU, etc).""")

def train_fn(step, batch, config, model, optimizer):
    model.train()
    x, y = batch['x'].cuda(), batch['y'].cuda()
    model_dict, loss_dict = {}, {}
    
    p_y_logits = model(x.flatten(1, -2))
    loss = F.cross_entropy(p_y_logits, y, reduction="mean")
    loss_dict["Loss"] = loss
    
    acc = (torch.argmax(p_y_logits, dim=-1) == y).type(torch.float32).mean()
    loss_dict["Acc"] = acc
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return model_dict, loss_dict

def eval_fn(step, config, model, dataloader, dataset):
    model.eval()
    loss_list, acc_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['x'].cuda(), batch['y'].cuda()
            p_y_logits = model(x.flatten(1, -2))
            loss = F.cross_entropy(p_y_logits, y, reduction="mean")
            loss_list.append(loss.detach().cpu())
            
            acc = (torch.argmax(p_y_logits, dim=-1) == y).type(torch.float32).mean()
            acc_list.append(acc.detach().cpu())
        
    eval_dict = {}
    eval_dict["Loss"] = np.mean(loss_list)
    eval_dict["Acc"] = np.mean(acc_list)
    return eval_dict

def save_fn(step, batch, model_dict, loss_dict, config):
    pass

def experiment(args):
    # Sync configs
    config = get_config(args.config.replace("default", "nturgbd_skeletons_transformer"))
    config = merge_args_into_config(args, config)
    config = Namespace(**config)
    print("Experiment: config loaded and merged.")
    
    if not args.dist or args.rank == 0:
        print("=" * 10 + "MERGED CONFIGS" + "=" * 10)
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(config)).items())))
        print("=" * 34)
    
    # Dataset
    train_dataset = NTURGBDSkeletons(args.data_dir, 
                                     eval_type=config.eval_type,
                                     split="train",
                                     transform=UniformSampling(config.num_frames))
    example = train_dataset.__getitem__(0)
    eval_dataset = NTURGBDSkeletons(args.data_dir,
                                    eval_type=config.eval_type,
                                    split="eval",
                                    transform=UniformSampling(config.num_frames))
    print("Experiment: dataset prepared.")
    
    # Model
    model = ClassificationTransformer(example["x"].shape[-1],
                                      seq_len=example["x"].flatten(0, -2).shape[0],
                                      num_classes=len(train_dataset.classes),
                                      embed_dim=config.embed_dim,
                                      depth=config.depth,
                                      num_heads=config.num_heads,
                                      mlp_ratio=config.mlp_ratio,
                                      qkv_bias=config.qkv_bias,
                                      drop_rate=config.drop_rate,
                                      attn_drop_rate=config.attn_drop_rate,
                                      drop_path_rate=config.drop_path_rate,
                                      norm_layer=config.norm_layer,
                                      act_layer=config.act_layer)
    print("Experiment: model prepared.")    
    model = model.cuda()
    
    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu_id]
        )
        sampler = data.DistributedSampler(train_dataset)
        lr = config.base_lr * args.world_size
        
        print("Experiment: distributed prepared.")
    else:
        sampler = None
        lr = config.base_lr
        
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=config.batch_size,
                                       sampler=sampler,
                                       num_workers=len(os.sched_getaffinity(0)))
    eval_dataloader = data.DataLoader(eval_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=len(os.sched_getaffinity(0)) // 2)
    print("Experiment: dataloader prepared.")
    
    # Optimizer
    lr_warmup_steps = config.lr_warmup_epochs * len(train_dataloader)
    lr_fn = lr_fn_warmup_cos(lr, lr_warmup_steps, config.epochs * len(train_dataloader))
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=lr,
    )
    lr_scheduler = CustomLRScheduler(lr_fn, optimizer)
    print("Experiment: optimizer prepared.")
    
    cudnn.benchmark = True
    
    ckpt_dict = {"model": model.module if args.dist else model,
                 "optimizer": optimizer}
    epoch, ckpt_fn = checkpoint_and_load(
        args.ckpt_dir,
        **ckpt_dict
    )
    lr_scheduler(epoch * len(train_dataloader))
    print("Experiment: checkpointing prepared.")
    
    print(f"Experiment: beginning training [{epoch}/{config.epochs}].")
        
    train_fn_ = partial(train_fn, config=config, model=model, optimizer=optimizer)
    eval_fn_ = partial(eval_fn, config=config, model=model, dataloader=eval_dataloader, dataset=eval_dataset)
    # save_fn_ = partial(save_fn, args=args)
    
    summary_writer = SummaryWriter(args.summary_dir)
    trainer = EpochTrainer(train_fn=train_fn_,
                           model=model,
                           optimizer=optimizer,
                           dataloader=train_dataloader,
                           epochs=config.epochs,
                           print_freq=args.print_freq,
                           ckpt_fn=ckpt_fn,
                           summary_writer=summary_writer)
    
    trainer.reg_eval(eval_fn_,
                     args.eval_freq)
    # trainer.reg_save(save_fn_,
    #                  args.save_freq)
    trainer.reg_lr_scheduler(lr_scheduler)
    if args.dist:
        trainer.reg_rank(args.rank)
    
    trainer.train(epoch)

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.dist:
        set_distributed_mode(args)
    
    experiment_setup(args)
    experiment(args)
