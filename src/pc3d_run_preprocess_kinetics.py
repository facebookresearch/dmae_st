from pathlib import Path

from pc3d.main_preprocess_kinetics import get_args_parser, main
from pc3d_submitit_preprocess_kinetics import get_init_file, get_shared_folder

if __name__ == '__main__':
    args = get_args_parser()
    args.add_argument("--job_dir", type=str)
    args.add_argument("--openmm", action="store_true",
                      help="Use openmm script")
    args = args.parse_args()
    
    if args.job_dir:
        args.output_dir = args.job_dir
    
    args.output_dir = "experiments/pc3d/preprocess_kinetics/TEST"
    # args.dist_url = get_init_file().as_uri()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.dist_url = get_init_file().as_uri()
    
    if args.openmm:
        raise NotImplementedError("openmm not yet supported")
        
    main(args)
