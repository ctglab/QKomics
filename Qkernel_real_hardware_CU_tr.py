# Importing standard Qiskit libraries
#CHANGE Eroror mitigation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import *
# Importing standard Qiskit libraries and configuring account
from qiskit_ibm_runtime import QiskitRuntimeService, Options
import qiskit_ibm_runtime 
#from qiskit.utils import algorithm_globals
#Load feature maps
from qiskit.circuit.library import ZZFeatureMap,ZFeatureMap
from Kernels.src.ComputeUncompute_2 import ComputeUncompute_2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
#Load other libraries
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import json
import argparse

#Load parameters
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
# returns JSON object as a dictionary
params= json.load(f)

#Create output directory
output_dir=params['Data']["Output_dir"]
kernel_dir_b= output_dir+'/'+params['ft_map']["ft_map"]+'_'+params['ft_map']["ent_type"]+'/'
try:
    os.makedirs(kernel_dir_b)
except OSError:
    print("Creation of the directory %s failed. Directory already exists" % kernel_dir_b)
else:
    print("Successfully created the directory %s" % kernel_dir_b)
########################QUANTUM SESSION#########################################################
# Loading your IBM Quantum account(s)
print('Loading IBM Quantum account')
# Loading your IBM Quantum account(s)
service=QiskitRuntimeService(channel="ibm_quantum",token="79236d6c79212a19f624756d3159ff4df842f676670b41257e20c724564c32de79977c3dff09e568097402b8c0ffb0f565aa511a772f2f9d6a1631835ca183a1")
print('selecting backend')
backend=service.get_backend(params['Backend']["backend"])
#backend = service.least_busy(operational=True, simulator=False)
#service.least_busy(simulator=False,
                             #operational=True,
                             #min_num_qubits=5)#service.get_backend(params['Backend']["backend"])
print(backend)
target = backend.target
coupling_map = target.build_coupling_map()
print('FT map instance')
#Instance FTMAP
n_qubits=params['Backend']["n_qubits"]
reps=params['ft_map']["reps"]
if params['ft_map']["ft_map"]=='ZZ':
    ft_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
    ft_map.ent_type=params['ft_map']["ent_type"]
else:
    ft_map = ZFeatureMap(feature_dimension=n_qubits, reps=1)

ft_map.decompose(reps=1).draw('mpl',style="bw",cregbundle=False,fold=-1,initial_state=True,
                              filename='images/{}_{}.png'.format(params['ft_map']["ft_map"],params['ft_map']['ent_type']))

#ft_map.draw(output='mpl')
#transpile circuit
print('transpile circuit')
pm=generate_preset_pass_manager(optimization_level=1,target=backend.target,initial_layout=[0,1,2,3],layout_method='trivial',seed_transpiler=42)
ft_map_t_qs = pm.run(ft_map)
ft_map_t_qs.draw('mpl',style='iqp', idle_wires=False,
                filename='images/{}_{}_transpiled_isa.png'.format(params['ft_map']["ft_map"],params['ft_map']['ent_type']))


######################## DATA PREPROCESSING ######################################################



# load data and sample
data_input = pd.read_csv(params['Data']["Input_file"], sep = ",")
data_input=data_input.sample(n=params['Data']["Sampling_size"],axis=0,random_state=42)

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
#labels = 'IntClustMemb'
#@TODO: Change label to the correct one
labels='label'

#Preprocess according to task
data_dict={}
print('create X_train',flush=True)
if params['Data']["task"]=='Unsupervised':
   y_train=data_input[labels]
   X_train=data_input[features]
else:
    print('Sorry this script is for Unsupervised learning')
   
#########################LAUNCH EXP#########################################################

#Set primitive sampler options
options = Options()
#Error mitigation level (resilience_level)
options.resilience_level = params['Backend']["resilience_level"]
#Optimization level
options.optimization_level = 3
#Number of shots
options.execution.shots = params['Backend']["shots"]
#Skip translation since the circuit is already transpiled
options.skip_transpilation= False

# Create a quantum kernel based on the transpiled feature map
#Set Primitive sampler
sampler = qiskit_ibm_runtime.Sampler(backend=backend, options=options)
#Set fidelity
fidelity = ComputeUncompute_2(sampler=sampler,pm=pm)
#Set kernel
qkernel = FidelityQuantumKernel(feature_map=ft_map,fidelity=fidelity,max_circuits_per_job=100)
# Iterate over the bandwidth values in params['Scaling']['bandwidth']
for i in params['Scaling']['bandwidth']:
    print(i)
    # Scale the features using MinMaxScaler with a feature range of (0, i*np.pi)
    scaler = MinMaxScaler(feature_range=(0, i*np.pi))
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    print(X_train_scaled.shape)
    # Evaluate the quantum kernel using the scaled features as inputs
    qk = qkernel.evaluate(X_train_scaled, X_train_scaled)
    print(qk.shape)
    print('Kernel evaluated')
    # Save the evaluated kernel to a pickle file
    with open(kernel_dir_b+'qk_tot_{}.pickle'.format(i), 'wb') as f:
        pickle.dump(qk, f)
    print('Kernel saved')
