# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# %%
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data

seed = 12345
algorithm_globals.random_seed = seed


# %%
xlr = pd.read_csv("/CTGlab/home/elia/qiskit test/dataset/test_xlr.txt", sep = "\t")
nsd = pd.read_csv("/CTGlab/home/elia/qiskit test/dataset/test_nsd.txt", sep = "\t")
sd = pd.read_csv("/CTGlab/home/elia/qiskit test/dataset/test_sd.txt", sep = "\t")

# %%
all_columns = ['MeanCvg', 'NRC_poolNorm', 'Class']
features = all_columns[:-1]
labels = all_columns[-1]

X_train, X_test, y_train, y_test = train_test_split(xlr[features], xlr[labels], train_size=500,test_size=500,random_state=89,stratify=xlr[labels])

# %%
scaler = MinMaxScaler(feature_range = (0, 2*np.pi))
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
nsd_scaled = scaler.transform(nsd[features])
sd_scaled = scaler.transform(sd[features])

y_nsd = nsd[labels]
y_sd = sd[labels]

# %%
feature_map = ZZFeatureMap(feature_dimension=len(features), reps=2, entanglement="linear")

adhoc_backend = QuantumInstance(
    BasicAer.get_backend("qasm_simulator"), shots=1024, seed_simulator=seed, seed_transpiler=seed)

adhoc_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=adhoc_backend)

# %%
normal_svc = SVC(kernel = "rbf", C= 20)
time_classic=datetime.now()
normal_svc.fit(X_train_scaled, y_train)
print('tempo esecuzione:', datetime.now()-time_classic)


# %%
'''
xlr_eval, _, y_xlr_eval, _ = train_test_split(X_test_scaled, y_test, train_size= 500, random_state=89,stratify=y_test)
nsd_eval, _, y_nsd_eval, _ = train_test_split(nsd_scaled, y_nsd, train_size= 500, random_state=89,stratify=y_nsd)
sd_eval, _, y_sd_eval, _ = train_test_split(sd_scaled, y_sd, train_size= 500, random_state=89,stratify=y_sd)

# %%
normal_score_xlr = normal_svc.score(xlr_eval, y_xlr_eval)
normal_score_nsd = normal_svc.score(nsd_eval, y_nsd_eval)
normal_score_sd = normal_svc.score(sd_eval, y_sd_eval)
print(f"RBF kernel classification test score XLR: {normal_score_xlr}")
print(f"RBF kernel classification test score NSD: {normal_score_nsd}")
print(f"RBF kernel classification test score SD: {normal_score_sd}")
'''
# %%
#evaluate kernel
time_quntum=datetime.now()
qkernel_train=adhoc_kernel.evaluate(X_train_scaled)
print('tempo esecuzione quantum train:', datetime.now()-time_quntum)
qkernel_test=adhoc_kernel.evaluate(X_test_scaled,X_train_scaled)
#substitute
adhoc_svc = SVC(kernel="precomputed",C=1)


# %%
time_quntum=datetime.now()
adhoc_svc.fit(qkernel_train, y_train)
print('tempo esecuzione quantum:', datetime.now()-time_quntum)


# %% [markdown]
# adhoc_score_xlr = adhoc_svc.score(xlr_eval, y_xlr_eval)
'''
adhoc_score_nsd = adhoc_svc.score(nsd_eval, y_nsd_eval)
adhoc_score_sd = adhoc_svc.score(sd_eval, y_sd_eval)
# 
# print(f"Callable kernel classification test score XLR: {adhoc_score_xlr}")
print(f"Callable kernel classification test score NSD: {adhoc_score_nsd}")
print(f"Callable kernel classification test score SD: {adhoc_score_sd}")
'''
# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

y_test_pred=adhoc_svc.predict(qkernel_test)
print ('accuracy score: %0.3f' % accuracy_score(y_test, y_test_pred))
C=confusion_matrix(y_true=y_test,y_pred=y_test_pred)
print(C)

# %%
'''
kernel = adhoc_kernel.evaluate(x_vec = X_train_scaled)

# %%
adhoc_matrix_train = adhoc_kernel.evaluate(x_vec=X_train_scaled)
adhoc_matrix_test = adhoc_kernel.evaluate(x_vec=xlr_eval, y_vec=y_xlr_eval)


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(
     np.asmatrix(adhoc_matrix_train), interpolation="nearest", origin="upper", cmap="Blues")
axs[0].set_title("Ad hoc training kernel matrix")
axs[1].imshow(np.asmatrix(adhoc_matrix_test), interpolation="nearest", origin="upper", cmap="Reds")
axs[1].set_title("Ad hoc testing kernel matrix")
plt.savefig('./kernel.png')
# 

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = xlr_eval[:, 0], xlr_eval[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax,adhoc_svc , xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.savefig('./surf_decision.png')



# %%
'''
