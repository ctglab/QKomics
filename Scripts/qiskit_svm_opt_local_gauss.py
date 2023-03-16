# %%
import numpy as np
import pandas as pd
import argparse
import prepare_training as preptr
#import matplotlib.pyplot as plt
from datetime import datetime

# %%
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix


from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data

time_start=datetime.now()
###FUNZIONI########################
def Calcolo_kernel(X_train,X_test):
    qkernel_train=adhoc_kernel.evaluate(X_train)
    qkernel_test=adhoc_kernel.evaluate(X_test,X_train)
    return qkernel_train,qkernel_test

def get_best_svc(cv_metrics):
    best_acc = 0
    for item in cv_metrics:

        if item[1] > best_acc:
            best_c = item[0]
            best_acc = item[1]
            best_std = item[2]
        elif item[1] == best_acc:
            if item[2] < best_std:
                best_c = item[0]
                best_acc = item[1]
                best_std = item[2]

    return best_c, best_acc, best_std

ap=argparse.ArgumentParser()
ap.add_argument('-train','--train_size',default=500,required=False,help='training set size')
ap.add_argument('-test','--test_size',default=500,required=False,help='test set size')

args=vars(ap.parse_args())

###PARAMETRI###################################
seed = 12345
algorithm_globals.random_seed = seed

#Train and test sizes
tr_size=int(args['train_size']) #train
ts_size=int(args['test_size'])#test
#C ranges
Cheese= [0.1, 0.5, 1, 5, 10,100,1000]

###Data loading and processing###############################
xlr = pd.read_csv("/CTGlab/home/elia/qiskit test/dataset/test_xlr.txt", sep = "\t")

# %%
all_columns = ['MeanCvg', 'NRC_poolNorm', 'Class']
features = all_columns[:-1]
labels = all_columns[-1]

#Split train and test
#X_train, X_test, y_train, y_test = train_test_split(xlr[features], xlr[labels],
                                                    #train_size=tr_size,test_size=ts_size,
                                                    #random_state=89,stratify=xlr[labels])
xlr_output_dict= preptr.prepare_training(data=xlr, columns = all_columns, split_to_train = True, add_noise = True, 
                     train_samples = tr_size, test_samples= ts_size, mu = 0, sigma = 0.05, seed_value = 42)

X_train=xlr_output_dict['x_train_with_noise']
X_test=xlr_output_dict['x_test_with_noise']
y_train= xlr_output_dict['y_train_with_noise']
y_test=xlr_output_dict['y_test_with_noise']

#scale data
scaler = MinMaxScaler(feature_range = (0, 2*np.pi))
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
###SET Quantum kernel####################
feature_map = ZZFeatureMap(feature_dimension=len(features), reps=2, entanglement="linear")

adhoc_backend = QuantumInstance(
    BasicAer.get_backend("qasm_simulator"), shots=1024, seed_simulator=seed, seed_transpiler=seed)

adhoc_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=adhoc_backend)

###OPTIMIZATION#################################

#Set  CV Folds
skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)

#kernel evaluation of each fold
kernelz=[]
i=1
time_sum=0
time_quntum=datetime.now()
print('inizio calcolo kernelz...')
for train_index,test_index in skf.split(X_train_scaled,y_train):
    time_iter=datetime.now()
    print('inizio calcolo kernelz iterazione: {}'.format(i))
    X_tr,X_ts= X_train_scaled[list(train_index)], X_train_scaled[list(test_index)]
    y_tr,y_ts= y_train[list(train_index)], y_train[list(test_index)]
    qk_tr,qk_ts=Calcolo_kernel(X_tr,X_ts)
    time=datetime.now()-time_iter

    print('tempo esecuzione quantum train iterazione {}: {}'.format(i, time))
    kernelz.append((qk_tr,qk_ts,y_tr, y_ts))
    i+=1
tot=datetime.now()-time_quntum
print('Total time kernel eval: {} \n'.format(tot))

#CROSS VALIDATION LOOP

cv_metrics = []
time_cv=datetime.now()
for curr_c in Cheese:
    curr_svc =SVC(kernel='precomputed', C = curr_c)
    curr_accuracies = []
    for item in kernelz:
        train_kernel = item[0]
        test_kernel = item[1]
        train_labels = item[2]
        test_labels = item[3]
        curr_svc.fit(train_kernel, train_labels)
        pred_labels = curr_svc.predict(test_kernel)
        #metrics calc
        curr_accuracies.append(accuracy_score(test_labels, pred_labels))

    curr_mean = round(np.mean(curr_accuracies ),4)
    curr_std = round(np.std(curr_accuracies ),4)
    print('C value: {} | Accuratezza media: {} | STD: {}'.format(curr_c, curr_mean, curr_std))
    cv_metrics.append( (curr_c, curr_mean, curr_std) )
time_cv=datetime.now()-time_cv
print('tempo esecuzione cv:{}'.format(time_cv))


best_c, best_acc, best_std=get_best_svc(cv_metrics=cv_metrics)
print('C value: {} | Accuratezza media: {} | STD: {}'.format(best_c, best_acc, best_std))
###TRAIN AND TEST WITH OPTIMIZED SVM ################
#evaluate kernel

time_quntum=datetime.now()
qkernel_train,qkernel_test=Calcolo_kernel(X_train_scaled,X_test_scaled)
print('tempo esecuzione calcolo kernelz totale:', datetime.now()-time_quntum)

#Define SVC
adhoc_svc = SVC(kernel="precomputed",C=best_c)



#Fit SVC
time_quntum=datetime.now()
adhoc_svc.fit(qkernel_train, y_train)
print('tempo fit qsvm:', datetime.now()-time_quntum)


#Test SVC
y_test_pred=adhoc_svc.predict(qkernel_test)
print ('accuracy score: %0.3f' % accuracy_score(y_test, y_test_pred))
C=confusion_matrix(y_true=y_test,y_pred=y_test_pred)
print('printing confusion matrix:')
print(C)
