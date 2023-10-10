#Load libraries
import numpy as np
import pandas as pd
import math
import threading
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
def Dummy_QKernel(items,iteration,data_dict,output_dir):
        for i in items:
            params=iteration[i]
            #Get params from iteration 

            ft=params[0]
            ent=params[1]
            case_=params [2]
            b=params[3]
            # Build ft_map
            #Build ft map
            feature_map=ft_maps_dict.get(ft)
            feature_map.entanglement=ent
            adhoc_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=adhoc_backend)
            #Get data 
            X_train=data_dict[case_]['X_train']
            X_test=data_dict[case_]['X_test']
            y_train=data_dict[case_]['y_train']
            y_test=data_dict[case_]['y_test']
            # Generate directory for kernel results

            kernel_dir_b= output_dir+case_+'/'+ft+'_'+ent+'/'
            try:
                os.makedirs(kernel_dir_b)
            except OSError:
                print ("Creation of the directory %s failed. Directory already exist" % kernel_dir_b)
            else:
                print ("Successfully created the directory %s " % kernel_dir_b)
            
            #scale data
            scaler = MinMaxScaler(feature_range = (0,b*np.pi))
            scaler.fit(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled = scaler.transform(X_train)
            print('#############{}_{}################'.format(ft,ent),flush=True)
            #Compute
            Compute_and_save_kernel(X_train_scaled,X_train_scaled,adhoc_kernel,kernel_dir_b,tag='tr_paral_{}'.format(b))
            time_k=datetime.now()-time_start
            print('Time employed for training and bandwidth {} :'.format(time_k))
            Compute_and_save_kernel(X_train_scaled,X_test_scaled,adhoc_kernel,kernel_dir_b,tag='ts_paral_{}'.format(b))
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
print(time_start)
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
            time_k=datetime.now()
            #Compute Training kernel
            qkernel_train=Compute_and_save_kernel(X_train=X_train_scaled,X_test=X_train_scaled,
                                                  adhoc_kernel=adhoc_kernel,dir=kernel_dir_b,tag='tr_{}'.format(b))
            time_k=datetime.now()-time_k
            print('Time employed for training and bandwidth {} :'.format(time_k))
            
            #Compute Test kernel
            qkernel_test=Compute_and_save_kernel(X_train=X_train_scaled,X_test=X_test_scaled,
                                                  adhoc_kernel=adhoc_kernel,dir=kernel_dir_b,tag='ts_{}'.format(b))
            print('Time employed for test and bandwidth {} :'.format(datetime.now()-time_k))
time_tot=datetime.now()-time_start    
print(datetime.now())       
print('Time total {} :'.format(time_tot))



        
### MULTI THREADING #########################################################################
num_thr=params['num_thr']
results_list = []
results_eigen = []
#iterable items
iteration=[]
for key in maps.keys():
    #get ft maps params
    ft=maps[key]['ft_map']
    ent=maps[key]['ent_type']

    for case_ in data_dict.keys():
        for b in bandwidth:
            iteration.append((ft,ent,case_,b))
    


inputs=np.arange(0,len(iteration),1).astype('int')

if num_thr > len(inputs):
    target_thrds = len(inputs)
else:
    target_thrds = num_thr
print(target_thrds)
#set and pair thread chunks and target                            
thread_chunk_size = math.floor(len(inputs)/ target_thrds)
target_lists = split(inputs, thread_chunk_size)

threads = []
thr = 1
time=datetime.now()
flag=0
#print('Computing Contribution entropy, the process migth take several hours :|')
for item in target_lists:
    threads.append(threading.Thread(target=Dummy_QKernel, 
                   args= (item,iteration,data_dict,output_dir)) )
    thr = thr+1
    flag+=1                    
for t in threads:
    t.start()
                        
for t in threads:
    t.join()
print('Time total with multiprocessing : {}'.format(datetime.now()-time))
    











