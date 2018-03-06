#!/usr/bin/env python
import sys
sys.path.append('../')
sys.path.append('../kde')
import kde.evaluate
import numpy
import scipy.integrate

kernels = ['bump',
           'cosine',
           'epanechnikov',
           'logistic', 
           'quartic',
           'tophat',
           'triangle',
           'tricube']


# Test kernel normalization in one dimensional euclidean space
training_points_1d = numpy.zeros(1)[numpy.newaxis,:]
for kernel in kernels:
    def f(x):
        x = numpy.array((x,))[numpy.newaxis, :]
        return kde.evaluate.estimate_pdf_brute(x, training_points_1d, kernel=kernel)[0]
    integral = scipy.integrate.quad(f, -20, 20)[0]
    if numpy.isclose(integral, 1):
        print("Test passed: 1D kernel '{:s}' integrates to {:f}".format(kernel, integral))
    else:
        print("TEST FAILED: 1D kernel '{:s}' integrates to {:f}.".format(kernel, integral))



# Test kernel normalization in two dimensional euclidean space
training_points_2d = numpy.zeros(2)[numpy.newaxis,:]
def lb(x):
    return -10
def ub(x):
    return 10

for kernel in kernels:
    def f(x, y):
        x = numpy.array((x,y))[numpy.newaxis, :]
        return kde.evaluate.estimate_pdf_brute(x, training_points_2d, kernel=kernel)[0]
    integral = scipy.integrate.dblquad(f, -10, 10, lb, ub)[0]
    if numpy.isclose(integral, 1):
        print("Test passed: 2D kernel '{:s}' integrates to {:f}".format(kernel, integral))
    else:
        print("TEST FAILED: 2D kernel '{:s}' integrates to {:f}.".format(kernel, integral))
