#!/bin/bash
set -e

filename=$1
groupname=${filename::-3} # assumes ".py" ending
experimentname=$2
args=${@:3}

d=`date +%Y-%m-%d`
j_name="$groupname/${d}:$experimentname"
conda_env=torch


output_dir=experiments/$j_name
sbatch_file="#!/bin/bash
#SBATCH --job-name=$j_name
#SBATCH --open-mode=append
#SBATCH --output=$output_dir/_slurm/slurm_logs/%j-%N.out
#SBATCH --error=$output_dir/_slurm/slurm_logs/%j-%N.err
#SBATCH --partition=a100
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1

source activate /data/home/$USER/miniconda/envs/${conda_env}
srun --label -u --open-mode=append bash $output_dir/_slurm/RUNSCRIPT.sh
"

run_script="
echo \$dist_url
if (( \$SLURM_PROCID > 0 )); then
    sleep 10;
fi
python $output_dir/src/$filename \\
--dir=$output_dir \\
--dist \\
$args
"

# ======================================================= #
mkdir -p $output_dir

mkdir $output_dir/_slurm
mkdir $output_dir/_slurm/slurm_logs
cp "$0" "$output_dir/_slurm/LAUNCH.sh"

mkdir $output_dir/src
cp -r src $output_dir

echo "$@" > $output_dir/_slurm/BASH.sh
echo "$sbatch_file" > $output_dir/_slurm/SBATCH.sh
echo "$run_script" > $output_dir/_slurm/RUNSCRIPT.sh
sbatch $output_dir/_slurm/SBATCH.sh