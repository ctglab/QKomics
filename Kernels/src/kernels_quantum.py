#############SCRIPT DESCRIPTION########################################################
# Quantum kernel for more qubits encoding metabric cna and exp pca features
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