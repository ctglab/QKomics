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
from qiskit_algorithms.state_fidelities import ComputeUncompute
#from Kernels.src.ComputeUncompute import ComputeUncompute
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



########################QUANTUM SESSION#########################################################
# Loading your IBM Quantum account(s)
print('Loading IBM Quantum account')
# Loading your IBM Quantum account(s)
service=QiskitRuntimeService(channel="ibm_quantum",token="95e70510a429e03c7cc4ce4c5c979c794135dbacffbebe5325dcd8b627cfe42c9f61e91945ba1adea992f81849c74cc5e0ed1ab93977db5084dd26a28fd25dd8")
print('selecting backend')
#backend = service.get_backend('ibmq_qasm_simulator')
backend=service.least_busy(operational=True, simulator=False)
#service.least_busy(simulator=False,
                             #operational=True,
                             #min_num_qubits=5)#service.get_backend(params['Backend']["backend"])
print(backend)
target = backend.target
coupling_map = target.build_coupling_map()
print('FT map instance')
#Instance FTMAP
n_qubits=4
reps=1
ft_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)

#transpile circuit
print('transpile circuit')
pm=generate_preset_pass_manager(backend=backend,optimization_level=1)
print(type(pm))
ft_map_t_qs = pm.run(ft_map)
######################## DATA PREPROCESSING ######################################################



#Generate sample data
X_train=np.random.rand(10,4)

scaler = MinMaxScaler(feature_range=(0, 1*np.pi))
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
   
#########################LAUNCH EXP#########################################################

#Set primitive sampler options
options = Options()
#Error mitigation level (resilience_level)
options.resilience_level = 1
#Optimization level
options.optimization_level = 3
#Number of shots
options.execution.shots = 2000
#Skip translation since the circuit is already transpiled
options.skip_transpilation= False



# Create a quantum kernel based on the transpiled feature map
#Set Primitive sampler
sampler = qiskit_ibm_runtime.Sampler(backend=backend, options=options)
#compose circuit and run
print(X_train_scaled[0])
print('Compose and run circuit with already transposed circuits')
ft_map_t_p=ft_map_t_qs.assign_parameters(X_train_scaled[0])
circuit=ft_map_t_qs.compose(ft_map_t_qs.inverse())
circuit.measure_active()
#run circuit
job = sampler.run(circuit,parameter_values=[np.random.rand(4)])
print(f">>> Job ID: {job.job_id()}")
print(f">>> Job Status: {job.status()}")


print('Case 2: Compose and run circuit and then transpose circuit')
circuit_2=ft_map.compose(ft_map.inverse())
circuit_2.measure_active()
#transpose
isa_circuit_2=pm.run(circuit_2)
job_2 = sampler.run(isa_circuit_2,parameter_values=[np.random.rand(4)])
print(f">>> Job ID: {job_2.job_id()}")
print(f">>> Job Status: {job_2.status()}")


#Run as fidelity kernel

print('Run as fidelity kernel normal way')
fidelity = ComputeUncompute(sampler=sampler)
#run circuit using fidelity
test=fidelity._run(ft_map_t_qs,ft_map_t_qs,X_train_scaled[0],X_train_scaled[1])

#Set fidelity
print('Run as a fidelity Kernel but with inside transpilation like case 2')
fidelity_2 = ComputeUncompute_2(sampler=sampler,pm=pm)
test_2=fidelity_2._run(ft_map,ft_map,X_train_scaled[0],X_train_scaled[1])

