# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import sys
import uuid
from pathlib import Path

from mae.main_finetune import get_args_parser
import submitit


def parse_args():
    trainer_parser = get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser])
    parser.add_argument("--job_dir", type=str, required=True,help="Job directory for the submitted slurm job.")
    
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
    
    ### Other args ###
    parser.add_argument("--signal_delay_s", type=int, default=120, help="Delay between kill signal and actual kill of the slurm job.")
    parser.add_argument("--account", type=str, default="all")
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoints/").is_dir():
        p = Path(f"/checkpoints/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.src_dir = sys.path[0]

    def __call__(self):
        sys.path.append(self.src_dir)
        
        import mae.main_finetune as trainer

        self._setup()
        
        trainer.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        
        # Setup checkpoint directory
        checkpoint_dir = Path(f"/checkpoints/{os.environ['USER']}/{str(job_env.job_id)}/checkpoints")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.args.checkpoint_dir = Path(os.path.join(str(self.args.output_dir), "checkpoints"))
        if self.args.rank == 0:
            self.args.checkpoint_dir.symlink_to(checkpoint_dir)


def main():
    args = parse_args()
    
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
    
    # Setup trainer args
    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted SLURMJOBID:[{job.job_id}] SLURMJOBNAME:[{name}]")


if __name__ == "__main__":
    main()
