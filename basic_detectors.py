#!/usr/bin/python

import numpy as np
import scipy
from statsmodels import robust
from statsmodels.tsa.seasonal import seasonal_decompose
from numpy.linalg import svd
import pandas

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

def tail_avg(x):
    """
    This is a utility function used to calculate the average of the last three
    datapoints in the series as a measure, instead of just the last datapoint.
    It reduces noise, but it also reduces sensitivity and increases the delay
    to detection.
    """
    try:
        t = (x[-1] + x[-2] + x[-3]) / 3
        return t
    except IndexError:
        return x[-1]


def median_absolute_deviation(x):
    """
    A timeseries is anomalous if the deviation of its latest datapoint with
    respect to the median is X times larger than the median of deviations.
    The median absolute deviation is a measure of statistical dispersion.
    Return X as a KPI (the dispersion of the data).
    """

    series = pandas.Series(x)
    median = series.median()
    demedianed = np.abs(series - median)
    median_deviation = demedianed.median()

    # The test statistic is infinite when the median is zero,
    # so it becomes super sensitive. We play it safe and skip when this happens.
    if median_deviation == 0:
        return False

    test_statistic = [x for x in demedianed][-1] / median_deviation

    return test_statistic

def z_score(x):

    series = x
    stdDev = scipy.std(series)
    mean = np.mean(series)
    tail_average = tail_avg(series)
    z_score = (tail_average - mean) / stdDev
    return z_score
 
def grubbs(x):

    series = x
    len_series = len(series)
    threshold = scipy.stats.t.isf(.05 / (2 * len_series), len_series - 2)
    threshold_squared = threshold * threshold
    grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))

    return grubbs_score



def TSD(x):
    result = seasonal_decompose(x, model='additive', freq=1440)
    return result.trend, result.seasonal, result.resid

def main():
    t = np.random.randint(1, 20, 20)
    print t
    begin = 0
    end = begin + 10
    while end < len(t):
        s = t[begin:end]
        #print median_absolute_deviation(s)
        print grubbs(s)
        begin += 1
        end += 1

main()

