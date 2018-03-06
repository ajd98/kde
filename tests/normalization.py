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

for kernel in kernels:
    if kernel in ['logistic', 'gaussian']:
        def ymin(x):
            return -10
        def ymax(x):
            return 10
        xmin = -10
        xmax = 10

    else:
        def ymin(x):
            return -1
        def ymax(x):
            return 1
        xmin = -1
        xmax = 1

    def f(x, y):
        x = numpy.array((x,y))[numpy.newaxis, :]
        return kde.evaluate.estimate_pdf_brute(x, training_points_2d, kernel=kernel)[0]

    integral = scipy.integrate.dblquad(f, xmin, xmax, ymin, ymax)[0]
    if numpy.isclose(integral, 1):
        print("Test passed: 2D kernel '{:s}' integrates to {:f}".format(kernel, integral))
    else:
        print("TEST FAILED: 2D kernel '{:s}' integrates to {:f}.".format(kernel, integral))

# Test kernel normalization in three dimensional euclidean space
training_points_3d = numpy.zeros(3)[numpy.newaxis,:]

for kernel in kernels:
    if kernel in ['logistic', 'gaussian']:
        xmin = -10
        xmax = 10

    else:
        xmin = -1
        xmax = 1

    def f(x, y, z):
        x = numpy.array((x,y,z))[numpy.newaxis, :]
        return kde.evaluate.estimate_pdf_brute(x, training_points_3d, kernel=kernel)[0]

    r = (xmin, xmax)
    integral = scipy.integrate.nquad(f, [r for i in range(3)])[0]
    if numpy.isclose(integral, 1):
        print("Test passed: 3D kernel '{:s}' integrates to {:f}".format(kernel, integral))
    else:
        print("TEST FAILED: 3D kernel '{:s}' integrates to {:f}.".format(kernel, integral))