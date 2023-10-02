#############SCRIPT DESCRIPTION########################################################
# Functions for Quantum kernel for Q-kernel computation on qiskit(0.42.1)
#Case all points
# Bandwidth opt
#
################################################START####################################
import numpy as np
import pandas as pd
import argparse
import os
import pickle
# From Qiskit
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel


def Compute_kernel(X_train,X_test,adhoc_kernel):
    """
    Compute kenel for given data and kernel type
    X_train=numpy array of data (train)
    X_test=numpy array of data(test )
    adhoc_kernel= QuantumKernel obj 
    """

    qkernel=adhoc_kernel.evaluate(X_test,X_train)
    return qkernel

def Save_kernel(qkernel,dir,tag):
    """
    Save kernel matrix as pickle
    qkernel=Numpy matrix 
    dir=string with directory to save qkernel as pickle
    tag=string with kernel specific tag
    """
    with open(dir+'qk_tot_{}.pickle'.format(tag),'wb') as f:
                        pickle.dump(qkernel, f)
    return 0

def Compute_and_save_kernel(X_train,X_test,adhoc_kernel,dir,tag):
        """
        """
        qkernel=Compute_kernel(X_train,X_test,adhoc_kernel)
        Save_kernel(qkernel,dir,tag)
        
        return 0
