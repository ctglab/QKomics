import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

from statistics import mean,variance
from sklearn.metrics.pairwise import rbf_kernel

def Compute_rbf_kernel(X_train,X_test):
    """
    """
    #get classical kernel
    n_ft=X_train.to_numpy().shape[1]
    gamma=1 / (n_ft * X_train.to_numpy().var())
    K_classic_tr = rbf_kernel(X_train,X_test, gamma = gamma)
    return(K_classic_tr)