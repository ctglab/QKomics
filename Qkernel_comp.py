#Load libraries
import numpy as np
import pandas as pd
import datetime 
import json
import argparse
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import BasicAer

from Kernels.src.kernels_quantum import Compute_kernel,Compute_and_save_kernel
from  Kernels.src.Preprocessing import Split_and_sample
 

ap=argparse.ArgumentParser()
ap.add_argument('-params','--parameters_file',
                default='hyper_param.json',
                required=False,
                help='json file with experiments info path')

args=vars(ap.parse_args())
params_dir=args['parameters_file']


############################PARAMETERS#########################################################

###########Load hyperparameters from json################

# Opening JSON file
f = open(params_dir) 
# returns JSON object as
# a dictionary
params= json.load(f)

#########Quantum Session parameters#############
seed=params['Backend']["seed"]
backend=params['Backend']["backend"]
n_qubits=params['Backend']["n_qubits"]
shots=params['Backend']["shots"]

algorithm_globals.random_seed = seed

#Set instance
adhoc_backend = QuantumInstance(
BasicAer.get_backend("qasm_simulator"), shots=shots, seed_simulator=seed, seed_transpiler=seed)

########################DATA PREPROCESSING ######################################################

input_file=params['Data']["Input_file"]
sampling_sz=params['Data']["Sampling_size"]
output_dir=params['Data']["Output_file"]

task=list(params['Data']["task"])[0]

# load data and sample
data_input = pd.read_csv(input_file, sep = ",")

#SELECT FT
features=[]
for i in range(1,n_qubits+1):
    name_cna='Component_'+str(i)+'_cna'
    name_exp='Component_'+str(i)+'_exp'
    features.append(name_cna)
    features.append(name_exp)
labels = 'IntClustMemb'


#Preprocess according to task
data_dict={}

if task=='Supervised':
    tr_sz=params['Data']["task"][task]['tr_sz']
    ts_sz=params['Data']["task"][task]['ts_sz']
    balanced=params['Data']["task"][task]['balanced']
    min_sz=params['Data']["task"][task]['min_sz']
    for case,class_ in params['Data']["task"][task[0]]['classes'].items():
        df_tot_sel=df_tot_sel=data_input.loc[data_input.IntClustMemb.isin(class_)]
        #
        X_train,y_train,X_test,y_test=Split_and_sample(df_tot_sel,
                                                       features,labels,
                                                       tr_sz=tr_sz,ts_sz=ts_sz,
                                                        min_sz=min_sz)
        data_dict[case]['X_train']=X_train
        data_dict[case]['X_test']=X_test
        data_dict[case]['y_train']=y_train
        data_dict[case]['X_test']=y_test
    
    











