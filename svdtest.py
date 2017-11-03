#!/usr/bin/python

import numpy as np
from numpy.linalg import svd
from numpy.linalg import norm

def construct_matrix():
    #return np.random.randint(1,1000,(10,10))
    # base part
    nrow = 10
    ncol = 5
    line = np.random.randint(20,90,(1,ncol))
    line = np.tile(line, nrow)
    matrix = line.reshape((nrow, ncol))
    # vaiable part
    p = np.random.randint(1,15, (nrow,ncol))
    matrix += p
    abnormal_row_index = np.random.randint(nrow)
    print 'abnormal_row_index: ', abnormal_row_index
    abnormal_row = np.random.randint(10,20, ncol)
    matrix[abnormal_row_index] += abnormal_row
    return matrix

def find_anomaly(x, appr_x):
    O = x - appr_x
    norms = [norm(line) for line in O]
    return np.argmax(norms)

def test_svd():
    x = construct_matrix()
    print "x:"
    print x
    nline, nrow = x.shape

    u, s, v = svd(x)
    n_singular_value = len(s)
    print "singular values:"
    print s

    print "number of singular values:", n_singular_value
    while n_singular_value > n_singular_value/2:
        appr_x = u[:,:n_singular_value].dot(np.diag(s[:n_singular_value])).dot(v[:n_singular_value, :])
        print "anomaly row: ", find_anomaly(x, appr_x)
        # if np.allclose(appr_x, x): print
        print "use singular value: %d\t norm: %10.8f" % (n_singular_value, np.linalg.norm(appr_x - x))
        n_singular_value -= 1


def main():
    test_svd()

main()
