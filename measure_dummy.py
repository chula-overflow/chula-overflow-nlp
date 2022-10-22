import numpy as np
from numpy.linalg import norm

def cosine_sim(a1, a2):
    """
    Input - ()Two vectors of dimension (1, features_dim)
    out - cosine sim of two vectors
    """
    return np.dot(a1, a2) / (norm(a1) * norm(a2))