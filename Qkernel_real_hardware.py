# Importing standard Qiskit libraries
#CHANGE Eroror mitigation
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import *
# Importing standard Qiskit libraries and configuring account
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Batch
import qiskit_ibm_runtime 
#Load fake backend
from qiskit.providers.fake_provider import FakeCairoV2
from qiskit.utils import algorithm_globals
#Load feature maps
from qiskit.circuit.library import ZZFeatureMap,ZFeatureMap
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel
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
# returns JSON object as
# a dictionary
params= json.load(f)

########################QUANTUM SESSION#########################################################
# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum",
                               token="ab57ee0cefb0e9dad544e4654f04acd4b277a9234129269b97ebebf6cd82e1b7a3d64e1f44006b01a1c3bb136589819486e6f408210e56bbbdd5cf795d056344")
backend = service.get_backend(params['Backend']["backend"])
target = backend.target
coupling_map = target.build_coupling_map()

#Instance FTMAP
n_qubits=params['Backend']["n_qubits"]
reps=params['ft_map']["reps"]
if params['ft_map']["ft_map"]=='ZZ':
    ft_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
    ft_map.ent_type=params['ft_map']["ent_type"]
else:
    ft_map = ZFeatureMap(feature_dimension=n_qubits, reps=1)

#ft_map.decompose(reps=1).draw('mpl',style="bw",cregbundle=False,fold=20,initial_state=True,)

#ft_map.draw(output='mpl')
#transpile circuit
ft_map_t_qs = transpile(ft_map,coupling_map=coupling_map,optimization_level=2,
                        initial_layout=params["Backend"]["initial_layout"],
                        seed_transpiler=42)
ft_map_t_qs.draw('mpl',style='bw', idle_wires=False,
                 filename='images/{}_transpiled.png'.format(params['ft_map']["ft_map"]))


######################## DATA PREPROCESSING ######################################################

output_dir=params['Data']["Output_dir"]

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
kernel_dir_b= output_dir+'/'+params['ft_map']["ft_map"]+'_'+params['ft_map']["ent_type"]+'/'
try:
    os.makedirs(kernel_dir_b)
except OSError:
    print("Creation of the directory %s failed. Directory already exists" % kernel_dir_b)
else:
    print("Successfully created the directory %s" % kernel_dir_b)

options = Options()
options.resilience_level = params['Backend']["resilience_level"]
options.optimization_level = 1
options.execution.shots = params['Backend']["shots"]
options.skip_translation = True

# Compute kernel
"""
with Session(backend=backend) as session:
    sampler = qiskit_ibm_runtime.Sampler(backend=backend, options=options, session=session)
    fidelity = ComputeUncompute(sampler=sampler)
    qkernel = FidelityQuantumKernel(feature_map=ft_map_t_qs, fidelity=fidelity)
"""

sampler = qiskit_ibm_runtime.Sampler(backend=backend, options=options)
fidelity = ComputeUncompute(sampler=sampler)
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
    qk = qkernel.evaluate(X_train_scaled, X_train_scaled)
    print(qk.shape)
    # Save the evaluated kernel to a pickle file
    with open(kernel_dir_b+'qk_tot_{}.pickle'.format(i), 'wb') as f:
        pickle.dump(qk, f)
    print('Kernel evaluated')
