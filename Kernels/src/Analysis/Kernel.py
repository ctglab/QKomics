import numpy as np
import pandas as pd
import argparse
import os


def Kernel_concentration(matrix):
    diagonal = np.diag(matrix)
    non_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)].flatten()  # Extract non-diagonal elements
    return np.var(non_diagonal)