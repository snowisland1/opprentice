import numpy as np
from statsmodels import robust

def MAD(a):
    """ calculate the Median Absolute Deviation """
    return robust.mad(a, c = 1.0)

def simple_ma(x, N):
    """ simple moving average """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def weighted_MA(x, N):
    """ weighted moving average """
    pass

def EWMA(x, N):
    """ exponential weighted moving average """
    pass


