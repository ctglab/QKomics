#Load libraries
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import argparse
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel


from sklearn.preprocessing import MinMaxScaler

from Kernels.src.kernels_quantum import Compute_kernel,Compute_and_save_kernel
from  Kernels.src.Preprocessing import Split_and_sample



### Function for paralellizing  ###
def Dummy_QKernel(X_train,X_test,adhoc_kernel,dir,b):
        scaler = MinMaxScaler(feature_range = (0,b*np.pi))
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        Compute_and_save_kernel(X_train,X_test,adhoc_kernel,dir,tag='tr_{}'.format(b))
        return 0

def split(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]

#######
 

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

######################## DATA PREPROCESSING ######################################################

input_file=params['Data']["Input_file"]
sampling_sz=params['Data']["Sampling_size"]
output_dir=params['Data']["Output_dir"]

task=list(params['Data']["task"])[0]

# load data and sample
data_input = pd.read_csv(input_file, sep = ",")
data_input=data_input.sample(n=sampling_sz,axis=0,random_state=42)


#SELECT FT
features=[]
for i in range(1,int(n_qubits/2) +1):
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
    for case,class_ in params['Data']["task"][task]['classes'].items():
        df_tot_sel=df_tot_sel=data_input.loc[data_input.IntClustMemb.isin(class_)]
        #
        X_train,y_train,X_test,y_test=Split_and_sample(df_tot_sel,
                                                       features,labels,
                                                       tr_sz=tr_sz,ts_sz=ts_sz,
                                                        min_sz=min_sz)
        data_dict[case]={}
        data_dict[case]['X_train']=X_train
        data_dict[case]['X_test']=X_test
        data_dict[case]['y_train']=y_train
        data_dict[case]['y_test']=y_test
        print('{} train = {} samples '.format(case,y_train.shape))
        print('{} test = {} samples '.format(case,y_test.shape))

#############COMPUTE QUANTUM KERNELS###################

###SET Quantum kernel####################

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
    adhoc_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=adhoc_backend)

    print('#############{}_{}################'.format(ft,ent),flush=True)
    
    
    for case_ in data_dict.keys():
        print('----------CASE {}-------'.format(case_),flush=True)
        X_train=data_dict[case_]['X_train']
        X_test=data_dict[case_]['X_test']
        y_train=data_dict[case_]['y_train']
        y_test=data_dict[case_]['y_test']
        
        # Generate directory for kernel results
        kernel_dir_b= output_dir+'/'+case_+'/'+ft+'_'+ent+'/'

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
            X_test_scaled = scaler.transform(X_test)
            print(X_train_scaled.shape)
            #Compute kernel
            time_start=datetime.now()
            #Compute Training kernel
            qkernel_train=Compute_and_save_kernel(X_train=X_train_scaled,X_test=X_train_scaled,
                                                  adhoc_kernel=adhoc_kernel,dir=kernel_dir_b,tag='tr_{}'.format(b))
            time_k=datetime.now()-time_start
            print('Time employed for training and bandwidth {} :'.format(time_k))
            
            #Compute Test kernel
            qkernel_test=Compute_and_save_kernel(X_train=X_train_scaled,X_test=X_test_scaled,
                                                  adhoc_kernel=adhoc_kernel,dir=kernel_dir_b,tag='ts_{}'.format(b))
            
time_tot=datetime.now()-time_start           
print('Time total {} :'.format(time_tot))

  











