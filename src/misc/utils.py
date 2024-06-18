import numpy as np
from enum import Enum

def mat2str(mat):
    return str(mat).replace("'",'"').replace('(','<').replace(')','>').replace('[','{').replace(']','}')  

def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])

def moving_average(a, n=3) :
    """
    Computes a moving average used for reward trace smoothing.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class FeatureExtractor(Enum):
    MLP = 0
    GCN = 1
    MPNN = 2


class RLAlgorithm(Enum):
    A2C = 0
    PPO = 1
    SAC = 2
