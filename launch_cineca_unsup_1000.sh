#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --partition=g100_usr_prod
#SBATCH --mem=100GB
##SBATCH --mem-per-cpu=


module purge

module load anaconda3;
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate dummy_env_old;

cd /g100/home/userexternal/vrepetto/Quantum-Machine-Learning-for-Expression-Data

python Qkernel_comp_unsup.py -params hyper_param_unsup_1000.json >stdoutput_unsupervised_1000_pt2.txt
##21 > slurm-out_2_cpu_20_vol 2> slurm-err_2