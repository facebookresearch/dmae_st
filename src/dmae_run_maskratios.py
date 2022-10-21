from pathlib import Path

from dmae.main_maskratios import get_args_parser, main
# from dmae_submitit_pretrain import get_init_file, get_shared_folder

if __name__ == '__main__':
    args = get_args_parser()
    args.add_argument("--job_dir", type=str)
    args = args.parse_args()
    
    if args.job_dir:
        args.output_dir = args.job_dir
    
    args.output_dir = "experiments/dmae-st/TEST"
    args.dist_url = "tcp://localhost:40001"
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
