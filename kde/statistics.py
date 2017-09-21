#!/usr/bin/env python
'''
Routines for computing weighted variance and weighted covariance matrices.
'''
import numpy

def weightedcovariance(m, w):
    '''
    Returns an unbiased estimate of the weighted covariance matrix for the 
    N by k matrix ``m`` where N is the number of samples and k is the number
    of dimensions of data per sample, and the corresponding length-N vector
    ``w`` representing the weight of each sample.

    >>> mat = numpy.array([[1,5,7],
                           [3,4,1],
                           [2,7,5],
                           [5,4,3]])
    >>> w = numpy.array([.2,.4,.1,.3])
    >>> weightedcovariance(mat, w)
        
    array([[ 2.98571429, -1.07142857, -2.6       ],
           [-1.07142857,  1.21428571,  1.85714286],
           [-2.6       ,  1.85714286,  7.37142857]])

    '''
    # Normalize weights
    w /= w.sum()
    # Weight each observation by its respective weight.
    wm = numpy.multiply(w,m.T).T 

    #Calculate the mean of each column
    avg = wm.sum(axis=0) 
    submat = m - avg 
    wsum = w.sum()
    inv_bias = wsum/(numpy.square(wsum) - w.dot(w))
    cov = inv_bias*submat.T.dot(numpy.multiply(w, submat.T).T) 
    return cov

def wvariance(m, w):
    '''
    Return the weighted variance for the vectory of samples ``m`` with 
    associated weights ``w``.
    '''
    wsum = w.sum()
    mean = (w*m).sum()/wsum 
    inv_bias = wsum/(numpy.square(wsum) - w.T.dot(w))
    cov = inv_bias * numpy.sum(w*numpy.square((m - mean)))
    return cov
