#Load libraries
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import argparse
from qiskit_algorithms.utils import algorithm_globals
#from qiskit import BasicAer
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler, BackendSampler
#from qiskit_ibm_runtime import Options
#mport qiskit_ibm_runtime 
#Load feature maps
from qiskit.circuit.library import ZZFeatureMap,ZFeatureMap
#from qiskit.algorithms.state_fidelities import ComputeUncompute
from Kernels.src.ComputeUncompute import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from sklearn.preprocessing import MinMaxScaler

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
print('Loading Parameters')
# Opening JSON file
f = open(params_dir) 
# returns JSON object as
# a dictionary
params= json.load(f)

#########Quantum Session parameters#############
backend=params['Backend']["backend"]
n_qubits=params['Backend']["n_qubits"]
shots=params['Backend']["shots"]

algorithm_globals.random_seed = 12345

#Set backend
backend = AerSimulator(method='automatic',max_parallel_threads=1, max_parallel_experiments=1)
print(backend.available_devices())

######################## DATA PREPROCESSING ######################################################

input_file=params['Data']["Input_file"]
sampling_sz=params['Data']["Sampling_size"]
output_dir=params['Data']["Output_dir"]

task=params['Data']["task"]

# load data and sample
data_input = pd.read_csv(input_file, sep = ",")
data_input=data_input.sample(n=min(sampling_sz,len(data_input)),axis=0,random_state=42)


#SELECT FT

features=[]
if params['Data']["encoding"]=='separated':
    for i in range(1,int(n_qubits/2) +1):
        name_cna='Component_'+str(i)+'_cna'
        name_exp='Component_'+str(i)+'_exp'
        features.append(name_cna)
        features.append(name_exp)
else:
    for i in range(1,int(n_qubits) +1):
        name_='Component_'+str(i)
        features.append(name_)
labels = 'IntClustMemb'



#Preprocess according to task
data_dict={}
print('create X_train',flush=True)
if task=='Unsupervised':
   y_train=data_input[labels]
   X_train=data_input[features]
else:
    print('Sorry this script is for Unsupervised learning')
   
print('X_train shape:',X_train.shape)
#############COMPUTE QUANTUM KERNELS###################

###SET Quantum kernel####################
print('Set Q kernel',flush=True)
n_reps=params['ft_maps']['n_reps']
maps=params['ft_maps']['maps']

feature_map_ZZ = ZZFeatureMap(feature_dimension=len(features), reps=n_reps)
feature_map_Z = ZFeatureMap(feature_dimension=len(features), reps=n_reps)
ft_maps_dict={'ZZ': feature_map_ZZ,
              'Z':feature_map_Z
              }

#Get scaling parameters
bandwidth=params['Scaling']['bandwidth']
time_start=datetime.now()
#Loop over cases,ft maps, and scaling
for key in maps.keys():
    #get ft maps params
    ft=maps[key]['ft_map']
    ent=maps[key]['ent_type']

    #Build ft map
    feature_map=ft_maps_dict.get(ft)
    feature_map.entanglement=ent
    sampler =BackendSampler(backend=backend,options={'shots':shots})
    #sampler=Sampler(options={'shots':shots})
    #Set fidelity
    fidelity = ComputeUncompute(sampler=sampler)
    #Set kernel
    qkernel= FidelityQuantumKernel(feature_map=feature_map,fidelity=fidelity)
    print('#############{}_{}################'.format(ft,ent),flush=True)
    kernel_dir_b= output_dir+'/'+ft+'_'+ent+'/'
    try:
        os.makedirs(kernel_dir_b)
    except OSError:
        print ("Creation of the directory %s failed. Directory already exist" % kernel_dir_b)
    else:
        print ("Successfully created the directory %s " % kernel_dir_b)
    print(kernel_dir_b)
    for b in bandwidth:
        print(b,flush=True)
        #scale#
        scaler = MinMaxScaler(feature_range = (0,b*np.pi))
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
       
        print(X_train_scaled.shape)
        #Compute kernel
        time_k=datetime.now()
        #Compute Training kernel
        qkernel_train=Compute_and_save_kernel(X_train=X_train_scaled,X_test=X_train_scaled,
                                              adhoc_kernel=qkernel,dir=kernel_dir_b,tag='{}'.format(b))
        time_k=datetime.now()-time_k
        print('Time employed for  bandwidth {} :'.format(time_k))
            
        
            
time_tot=datetime.now()-time_start 
       
print('Time total {} :'.format(time_tot))

  











