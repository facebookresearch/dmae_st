import os
import sys
import time
from argparse import Namespace
from datetime import datetime

from util.training.args import save_args
from util.training.checkpointing import move_newest_checkpoint
from util.training.logger import Logger


def experiment_setup(args: Namespace, print_args: bool=True):
    args.logdir = os.path.join(args.dir, "logs")     # Logging
    args.ckpt_dir = "{}/checkpoints".format(args.dir)     # Checkpointing
    args.save_dir = "{}/saves/training".format(args.dir)     # Saving training dir
    args.summary_dir = "{}/tfsummary".format(args.dir)     # Tensorboard / TFSummary dir
    
    # Prepare directories, checkpoints and save arguments
    if not args.dist or args.rank == 0:        
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.summary_dir, exist_ok=True)

        # Record args
        args_dir = os.path.join(args.dir, "args.json")
        save_args(args_dir, args)
    else: # when args.rank != 0
        time.sleep(5.) # sleep to allow for execution
    
    # Prepare logger(s)
    if args.dist:
        logger = Logger(os.path.join(args.logdir,
                                    "{}({})-{}.log".format(os.environ["SLURM_JOB_ID"],
                                                           args.rank,
                                                           datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))
    else:
        logger = Logger(os.path.join(args.logdir,
                                    "{}-{}.log".format(os.environ["SLURM_JOB_ID"],
                                                       datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))
    sys.stdout = logger
    
    if print_args:
        print("=" * 15 + "ARGS" + "=" * 15)
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
        print("=" * 34)
