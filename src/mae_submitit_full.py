# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import argparse
import os
from pprint import pprint
import sys
from typing import Iterable, List
import uuid
from pathlib import Path

import submitit
from importlib import import_module

def get_slurms_args_parser():
    parser = argparse.ArgumentParser("Submitit configurations and arguments")
    
    parser.add_argument("--job_dir", type=str, required=True, help="Job directory for the submitted slurm job.")
    
    parser.add_argument("--partition", type=str, required=True, help="Partition to submit to.")
    parser.add_argument("--time", type=int, default=0, help="Timeout in seconds. 0 is no timeout.")
    parser.add_argument("--comment", type=str, default="", help="Comment for slurm.")
    parser.add_argument("--exclude", type=str, help="Identifiers to exclude.")
    
    ### Resource args ###
    parser.add_argument("--gpus_per_task", type=int, default=1, help="GPUs per task. (Default: 1)")
    parser.add_argument("--cpus_per_task", type=int, default=12, help="CPUs per GPU.")
    
    parser.add_argument("--num_gpus", type=int, help="Total number of gpus per job.")
    
    parser.add_argument("--nodes", type=int, help="Number of nodes for the job.")
    parser.add_argument("--tasks_per_node", type=int, default=8, help="Tasks per node. (Default: 8)")
    
    parser.add_argument("--signal_delay_s", type=int, default=120, help="Delay between kill signal and actual kill of the slurm job.")
    parser.add_argument("--account", type=str, default="all")
    
    return parser

class Trainer(object):
    def __init__(self, args, exp_args):
        self.args = args
        self.exp_args = exp_args
        self.src_dir = sys.path[0]

    def __call__(self):
        sys.path.append(self.src_dir)
        
        # Run pretraining
        import mae.main_pretrain as trainer_pretrain
        pretrain_arglist = self._filter_arglist(self.exp_args, "PRETRAIN")
        pretrain_args = trainer_pretrain.get_args_parser().parse_args(pretrain_arglist)
        pretrain_args = self._setup(pretrain_args, "PRETRAIN")
        trainer_pretrain.main(pretrain_args)
        
        # Run finetuning
        import mae.main_finetune as trainer_finetune
        finetune_argslist = self._filter_arglist(self.exp_args, "FINETUNE")
        finetune_args = trainer_finetune.get_args_parser().parse_args(finetune_argslist)
        finetune_args = self._setup(finetune_args, "FINETUNE")
        finetune_args.finetune = os.path.join(pretrain_args.checkpoint_dir, f"checkpoint-{pretrain_args.epochs-1:05d}.pth")
        trainer_finetune.main(finetune_args)
        
        import mae.main_test as trainer_test
        test_argslist = self._filter_arglist(self.exp_args, "TEST")
        test_args = trainer_test.get_args_parser().parse_args(test_argslist)
        test_args = self._setup(test_args, "TEST")
        test_args.finetune = os.path.join(finetune_args.output_dir, f"best_checkpoint.pth")
        trainer_test.main(test_args)

    def _filter_arglist(self, arglist: Iterable[str], prefix: str):
        filtered_arglist = filter(lambda x: f"--{prefix}." in x, arglist)
        mapped_arglist = map(lambda x: x.replace(f"--{prefix}.", "--"), filtered_arglist)
        return mapped_arglist

    def _setup(self, args, prefix):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        args.output_dir = Path(str(args.output_dir).replace("%j", str(job_env.job_id)))
        args.log_dir = args.output_dir
        args.gpu = job_env.local_rank
        args.rank = job_env.global_rank
        args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        
        # Setup checkpoint directory
        checkpoint_dir = Path(f"/checkpoints/{os.environ['USER']}/{str(job_env.job_id)}/{prefix}-checkpoints")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        args.checkpoint_dir = Path(os.path.join(str(args.output_dir), "checkpoints"))
        if args.rank == 0:
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(os.path.dirname(args.checkpoint_dir), exist_ok=True)
            args.checkpoint_dir.symlink_to(checkpoint_dir)
        return args

def aws_get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoints/").is_dir():
        p = Path(f"/checkpoints/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(aws_get_shared_folder()), exist_ok=True)
    init_file = aws_get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def aws_main():
    # Args
    args, unused_args = get_slurms_args_parser().parse_known_args()
    
    assert ((args.nodes is not None) != (args.num_gpus is not None),
            f"One of --nodes ({args.nodes}) or --num_gpus ({args.num_gpus}) must be specified.")
    
    # Setup Executor
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=10)
    
    name = args.job_dir.split("/")[-1]
    
    if args.partition == "lowpri" and args.time == 0:
        print("No timeout specified for lowpri, setting to 1440.")
        args.time = 1440
    
    kwargs = {}
    if args.comment:
        kwargs["slurm_comment"] = args.comment
    if args.exclude:
        kwargs["slurm_exclude"] = "a100-st-p4d24xlarge-[{}]".format(args.exclude)
    if args.account:
        kwargs["slurm_account"] = args.account
    
    if args.num_gpus:
        kwargs["slurm_num_gpus"] = args.num_gpus
    elif args.nodes:
        kwargs["slurm_nodes"] = args.nodes
        kwargs["slurm_tasks_per_node"] = args.tasks_per_node
        
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_signal_delay_s=args.signal_delay_s,
        slurm_time=args.time,
        slurm_gpus_per_task=args.gpus_per_task,
        slurm_cpus_per_task=args.cpus_per_task,
        name=name,
        **kwargs
    )
    
    # add dist_urls
    unused_args.append(f"--PRETRAIN.dist_url={get_init_file().as_uri()}")
    unused_args.append(f"--FINETUNE.dist_url={get_init_file().as_uri()}")
    unused_args.append(f"--TEST.dist_url={get_init_file().as_uri()}")
    
    # add job_dirs
    unused_args.append(f"--PRETRAIN.output_dir={args.job_dir}")
    unused_args.append(f"--FINETUNE.output_dir={os.path.join(args.job_dir, 'finetune')}")
    unused_args.append(f"--TEST.output_dir={os.path.join(args.job_dir, 'test')}")

    trainer = Trainer(args, unused_args)
    # trainer()
    job = executor.submit(trainer)

    print(f"Submitted SLURMJOBID:[{job.job_id}] SLURMJOBNAME:[{name}]")
    
    
if __name__ == "__main__":
    aws_main()