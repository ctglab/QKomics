# Quantum enhanced stratification of Breast Cancer: exploring quantum expressivity for real omics data
Stratification  of tumour samples from molecular descriptors (gene expression and copy number alteration) with quantum Kernels
## Input data
Input data contains the METABRIC dataset reduced with UMAP and the notebook to see the dimentionality recuction step. 
## Experimental Results

### Kernel_Results
This folder contains all quantum kernel computed in this work
### Results
This folder contains clustering results extracted for all the computed kernels
## Packages Requirments and usage
To avoide problems with the env requirments follow these steps:

1. Create conda env with python 3.10

```
conda create -n <your_env_name> python==3.10
```

2. Activate env

```
conda activate <your_env_name>
```

3. Install all packages present in requirments.txt

```
pip install -r requirments.txt
```

### Usage

Launch noisless Quantum Kernel simulation:

```
python Qkernel_comp_unsup_simulation.py -params utils/hyper_param_unsup.json 

```

Launch QPU Quantum Kernel computation:

```
python Qkernel_real_hardware_CU_tr.py -params utils/Qkernel_real_hardware_CU_tr.py

```

Launch clustering of a given set of kernels and compute Silhouette scores

```
python Analysis_unsup.py -params utils/hyper_param_unsup_analysis.json

```
