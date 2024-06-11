#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=g100_usr_prod
##SBATCH --mem=200GB
#SBATCH --account=IscrC_ENEO2
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=valeria.repetto291@gmail.com

module purge

module load anaconda3;
source ~/.bashrc;
##source "$CONDA_PREFIX/etc/profile.d/conda.sh";
conda activate  /g100/home/userexternal/vrepetto/.conda/envs/qiskit;
conda list;

cd /g100/home/userexternal/vrepetto/Quantum-Machine-Learning-for-Expression-Data

python Qkernel_comp_unsup_new.py -params utils/hyper_param_unsup_500.json >Output_files/stdoutput_unsup_500.txt
##21 > slurm-out_2_cpu_20_vol 2> slurm-err_2