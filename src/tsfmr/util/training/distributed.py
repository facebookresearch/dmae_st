import sys
import torch
import torch.distributed as dist
import os
import re

def set_distributed_mode(args):
    """
    Set world_size and rank in args and set distributed mode.

    Args:
        args (_type_): args
    """
    
    is_slurm_job = "SLURM_JOB_ID" in os.environ
    
    if is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_TASKS_PER_NODE"][0])
        if not args.dist_url:
            node_list = os.environ["SLURM_JOB_NODELIST"]
            if "[" in node_list:
                node_name = os.environ["SLURM_JOB_NODELIST"].split("[")[0] + re.split("-|,",
                                                                                      os.environ["SLURM_JOB_NODELIST"].split("[")[1].split("]")[0])[0]
            else:
                node_name = node_list
            port = "40000" if not args.port else f"{args.port}"
            args.dist_url = "tcp://" + node_name + f":{port}"
    else:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        
    print(f"Distributed: initializing distributed mode. URL: {args.dist_url} world_size: {args.world_size} rank: {args.rank}")
    
    for _ in range(8):
        try:
            dist.init_process_group(
                backend="nccl",
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank
            )
        except:
            if is_slurm_job:
                split = args.dist_url.split(":")
                print(f"Distributed: is_slurm_job=True, trying next port: {int(split[2]) + 1}")
                args.dist_url = f"{split[0]}:{split[1]}:{int(split[2]) + 1}"
                continue
            else:
                print("Distributed: init_process_group failed.")
                break
        args.gpu_id = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu_id)
        dist.barrier()
        return
    sys.exit("Distributed failed to start.")
