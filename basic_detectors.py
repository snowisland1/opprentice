#!/usr/bin/python

import numpy as np
import scipy
from statsmodels import robust
from statsmodels.tsa.seasonal import seasonal_decompose
from numpy.linalg import svd
import pandas

import sys

from sklearn.ensemble import GradientBoostingClassifier

ALGORITHMS = [
"median_absolute_deviation",
"z_score",
"grubbs",
"stddev_from_average",
"stddev_from_ewma",
"mean_subtraction_cumulation",
"histogram_bins",
]

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

def TSD(x):
    result = seasonal_decompose(x, model='additive', freq=1440)
    return result.trend, result.seasonal, result.resid


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

def stddev_from_average(x):
    """ (the average of the three datapoints) - (moving average) / (stddev) """
    series = pandas.Series(x)
    mean = series.mean()
    stdDev = series.std()
    t = tail_avg(x)

    return abs(t - mean) / stdDev

def stddev_from_ewma(x):
    series = pandas.Series(x)
    expAverage = pandas.stats.moments.ewma(series, com=50)
    #Series.ewm(ignore_na=False,min_periods=0,adjust=True,com=50).mean()
    stdDev = pandas.stats.moments.ewmstd(series, com=50)

    latest_expAverage = [a for a in expAverage][-1]
    latest_stdDev = [a for a in stdDev][-1]
    latest_elem = x[-1]

    return abs(latest_elem - latest_expAverage) / latest_stdDev


def mean_subtraction_cumulation(x):
    series = pandas.Series(x)
    series = series - series[0:len(series) - 1].mean()
    stdDev = series[0:len(series) - 1].std()

    return abs(x[-1]) / stdDev

def histogram_bins(x):
    series = scipy.array(x)
    t = tail_avg(x)
    bin_sizes, bin_edges = np.histogram(series)
    for i,e in enumerate(bin_edges):
        if t < e:
            break
    return float(bin_sizes[i-1])/float(len(x))

def train_model(X, Y):
    model = GradientBoostingClassifier()
    #base_model = lgb.LGBMClassifier(objective='binary',
    #                         boosting_type='gbdt',
    #                         num_leaves=28,
    #                         learning_rate=0.05,
    #                         bagging_fraction=0.8,
    #                         bagging_freq=5,
    #                         n_estimators=30,
    #                         nthread=5)
    model.fit(X, Y)
    return model

def readinput():
    filename = sys.argv[1]
    in_data = []
    out_data = []
    for line in open(filename):
        daytime, flow_in, flow_out = line.strip().split(",")
        in_data.append(int(flow_in))
        out_data.append(int(flow_out))

    return in_data, out_data


def detect_anomaly(t):
    begin = 0
    end = begin + 10
    # data
    X = []
    # label
    Y = []
    while end < len(t):
        s = t[begin:end]
        x = [globals()[algorithm](s) for algorithm in ALGORITHMS]
        print x
        X.append(x)
        Y.append(np.random.randint(0,2))
        begin += 1
        end += 1
    train_model(X, Y)

def main():
    #print ALGORITHMS
    #t = np.random.randint(1, 20, 40)
    #t = np.append(t, np.random.randint(20, 90, 3))
    #print t
    in_data, out_data = readinput()
    detect_anomaly(in_data)


main()

