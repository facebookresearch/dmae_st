# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from pathlib import Path

from dmae.main_masked_linprobe import get_args_parser, main
from mae_submitit_pretrain import get_init_file, get_shared_folder

if __name__ == '__main__':
    args = get_args_parser()
    args.add_argument("--job_dir", type=str)
    args = args.parse_args()
    
    if args.job_dir:
        args.output_dir = args.job_dir
    
    args.output_dir = "experiments/mae-st/TEST"
    args.dist_url = get_init_file().as_uri()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
