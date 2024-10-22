import numpy as np

def manual_field_transformation(U, alpha=0.1):
    return U + alpha * np.sin(U) + alpha * np.random.uniform(-1, 1, U.shape)