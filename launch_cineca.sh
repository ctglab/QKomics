#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH --nodes=2
#SBATCH --partition=g100_all_serial
##SBATCH --mem=200GB
##SBATCH --mem-per-cpu=


module purge

module load anaconda3;
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate dummy_env_old;

cd /g100/home/userexternal/vrepetto/Quantum-Machine-Learning-for-Expression-Data

python Qkernel_comp_parallel.py -params hyper_param.json >stdoutput_trial_multiproc.txtx
##21 > slurm-out_2_cpu_20_vol 2> slurm-err_2