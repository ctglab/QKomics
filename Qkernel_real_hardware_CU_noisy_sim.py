# Importing standard Qiskit libraries
#CHANGE Eroror mitigation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import FakeMumbai
from qiskit.visualization import *
from qiskit_aer.primitives import Sampler
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
# Importing standard Qiskit libraries and configuring account
from qiskit_ibm_runtime import QiskitRuntimeService, Options
#import qiskit_ibm_runtime 
#from qiskit.utils import algorithm_globals
#Load feature maps
from qiskit.circuit.library import ZZFeatureMap,ZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
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
from datetime import datetime

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
service=QiskitRuntimeService(channel="ibm_quantum",token="95e70510a429e03c7cc4ce4c5c979c794135dbacffbebe5325dcd8b627cfe42c9f61e91945ba1adea992f81849c74cc5e0ed1ab93977db5084dd26a28fd25dd8")
print('selecting backend')

#noise_model=NoiseModel.from_backend(params['Backend']["backend"])
backend_orig= service.get_backend(params['Backend']["backend"])
noise_model=NoiseModel.from_backend(backend_orig)
target=backend_orig.target
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
pm=generate_preset_pass_manager(optimization_level=3,coupling_map=coupling_map,initial_layout=[0,1,2,3],layout_method='trivial',seed_transpiler=42)

ft_map_t_qs = pm.run(ft_map)
ft_map_t_qs.draw('mpl',style='iqp', idle_wires=False,
                filename='images/{}_{}_transpiled.png'.format(params['ft_map']["ft_map"],params['ft_map']['ent_type']))


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
labels = 'IntClustMemb'

#Preprocess according to task
data_dict={}
print('create X_train',flush=True)
if params['Data']["task"]=='Unsupervised':
   y_train=data_input[labels]
   X_train=data_input[features]
else:
    print('Sorry this script is for Unsupervised learning')
   
#########################LAUNCH EXP#########################################################
print('Launch experiment')
print('Set Sampler')
sampler = Sampler(backend_options={"noise_model": noise_model})
#Set fidelity
print('Set fidelity')
fidelity = ComputeUncompute(sampler=sampler)
#Set kernel
print('Set kernel')
qkernel = FidelityQuantumKernel(feature_map=ft_map_t_qs,fidelity=fidelity)
# Iterate over the bandwidth values in params['Scaling']['bandwidth']
for i in params['Scaling']['bandwidth']:
    print(i)
    # Scale the features using MinMaxScaler with a feature range of (0, i*np.pi)
    scaler = MinMaxScaler(feature_range=(0, i*np.pi))
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    print(X_train_scaled.shape)
    # Evaluate the quantum kernel using the scaled features as inputs
    t=datetime.now()
    qk = qkernel.evaluate(X_train_scaled, X_train_scaled)
    print(qk.shape)
    print('Kernel evaluated')
    print('Execution time:',datetime.now()-t)
    # Save the evaluated kernel to a pickle file
    with open(kernel_dir_b+'qk_tot_{}.pickle'.format(i), 'wb') as f:
        pickle.dump(qk, f)
    print('Kernel saved')
