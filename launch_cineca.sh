#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --partition=g100_all_serial
##SBATCH --mem=10GB
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

python Qkernel_comp_unsup_new.py -params utils/hyper_param_unsup.json >Output_files/stdoutput_unsup_test_cpu_1_200.txt
##21 > slurm-out_2_cpu_20_vol 2> slurm-err_2