# %%
import numpy as np
from qiskit import BasicAer
#from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit import IBMQ

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data



from datetime import datetime

seed = 12348


# Loading your IBM Quantum account(s)


# %%
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd


# %%
f=open('log.txt','w')
#set backend
IBMQ.save_account('ab57ee0cefb0e9dad544e4654f04acd4b277a9234129269b97ebebf6cd82e1b7a3d64e1f44006b01a1c3bb136589819486e6f408210e56bbbdd5cf795d056344',overwrite=True)
provider = IBMQ.load_account()
IBMQ.providers()
#provider = IBMQ.get_provider('ibm-q','open','main')
provider = IBMQ.get_provider('partner-cnr','iit','qml-for-genomics')

num_qubits = 2

from qiskit.providers.ibmq import least_busy
possible_devices = provider.backends(filters=lambda x: 
        x.configuration().n_qubits >= num_qubits
                                       and 
        x.configuration().simulator == False)
backend = least_busy(possible_devices)
print(backend)


# %%
#Load training data

xlr = pd.read_csv("/CTGlab/home/elia/qiskit test/dataset/test_xlr.txt", sep = "\t")
nsd = pd.read_csv("/CTGlab/home/elia/qiskit test/dataset/test_nsd.txt", sep = "\t")
sd = pd.read_csv("/CTGlab/home/elia/qiskit test/dataset/test_sd.txt", sep = "\t")

# %%
all_columns = ['MeanCvg', 'NRC_poolNorm', 'Class']
features = all_columns[:-1]
labels = all_columns[-1]

X_train, X_test, y_train, y_test = train_test_split(xlr[features], xlr[labels], train_size=500, random_state=42,stratify=xlr[labels])

# %%
print('loading dataset')
scaler = MinMaxScaler(feature_range = (0, 2*np.pi))
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

nsd_scaled = scaler.transform(nsd[features])
sd_scaled = scaler.transform(sd[features])

y_nsd = nsd[labels]
y_sd = sd[labels]

# %% [markdown]
# ## Quantum kernel with sklearn.SVC

# %%
#define kernel and quantumInstance
print('define kernel and quantumInstance')
num_qubits = 2
shots= 1024
feature_map = ZZFeatureMap(feature_dimension=num_qubits,entanglement='linear')

quantum_instance = QuantumInstance(backend,shots=shots,skip_qobj_validation=False)

adhoc_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)#Creation of k



# %%
adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)

# %% [markdown]
# # SVC Optimization
# 
# SVC optimization with Halving algorithm

# %%
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

# %%


# %%
range_C=[x *0.01 for x in range(1,10)]
print(range_C)

param_grid = {"C":range_C,
              "tol":[1e-2] }
search = HalvingGridSearchCV(adhoc_svc, param_grid, resource='n_samples' ,max_resources='auto',random_state=0,refit=True,verbose=10).fit(X_train_scaled, y_train)

# %%
print('kernel_opt')
svc_opt=search.best_estimator_
print(svc_opt)
print(search.best_params_)
print(svc_opt,file=f)
print(search.best_params_,file=f)


# %% [markdown]
# time_quntum=datetime.now()
# adhoc_svc.fit(X_train_scaled, y_train)
# print('tempo esecuzione fit quantum:', datetime.now()-time_quntum)
# 

# %%
import pickle

def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


bk = {}
for k in dir():
    obj = globals()[k]
    if is_picklable(obj):
        try:
            bk.update({k: obj})
        except TypeError:
            pass

# to save session
with open('./After_fit_500pt_opt_bk.pkl', 'wb') as f2:
    pickle.dump(bk, f2)


# %%
xlr_eval, _, y_xlr_eval, _ = train_test_split(X_test_scaled, y_test, train_size= 500, random_state=42,stratify=y_test)
nsd_eval, _, y_nsd_eval, _ = train_test_split(nsd_scaled, y_nsd, train_size= 500, random_state=42,stratify=y_nsd)
sd_eval, _, y_sd_eval, _ = train_test_split(sd_scaled, y_sd, train_size= 500, random_state=42,stratify=y_sd)

# %%

# %%
time_xlr=datetime.now()
y_pred = svc_opt.predict(xlr_eval)

qsvc_score_xlr=accuracy_score(y_xlr_eval,y_pred) 
print(f"Callable kernel classification test score XLR: {qsvc_score_xlr}")
print('tempo di esecuzione valutazione xlr:',datetime.now()-time_xlr)

print(f"Callable kernel classification test score XLR: {qsvc_score_xlr}",file=f)


# %%
conf_matrix=confusion_matrix(y_xlr_eval,y_pred)
print(conf_matrix,file=f)

# %%
bk = {}
for k in dir():
    obj = globals()[k]
    if is_picklable(obj):
        try:
            bk.update({k: obj})
        except TypeError:
            pass

# to save session
with open('./After_classification_xlr_500pt__opt_bk.pkl', 'wb') as f2:
    pickle.dump(bk, f2)


# %% [markdown]
# ### Figures

# %%
import matplotlib.pyplot as plt
def make_meshgrid(x, y, h=0.3):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out, Z


fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface Quantum SVC ZZ feature map')
# Set-up grid for plotting.
X0, X1 = xlr_eval[:, 0], xlr_eval[:, 1]
xx, yy = make_meshgrid(X0, X1)

#plot_contours(ax,adhoc_svc , xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
Z = adhoc_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
out = ax.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.colorbar(out)
ax.scatter(X0, X1, c=y_xlr_eval, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('NRC_poolNorm ')
ax.set_xlabel('meanCvg  ')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()
plt.savefig('./decision_surface_ZZ_ftmap_500pt_opt.png')

# %%
bk = {}
for k in dir():
    obj = globals()[k]
    if is_picklable(obj):
        try:
            bk.update({k: obj})
        except TypeError:
            pass

# to save session
with open('./After_fig_500pt_opt_bk.pkl', 'wb') as f2:
    pickle.dump(bk, f2)



