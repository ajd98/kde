#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('../')
sys.path.append('../kde')
import kde.evaluate
import numpy
import scipy.integrate
from warningcolors import terminalcolors

kernels = ['bump',
           'cosine',
           'epanechnikov',
           'gaussian',
           'logistic', 
           'quartic',
           'tophat',
           'triangle',
           'tricube']

compactkernels = kde.evaluate.COMPACT_KERNELS

def main():
    # Test kernel normalization in one dimensional euclidean space
    print("Testing kernel normalization in 1D Euclidean space")
    training_points_1d = numpy.zeros(1)[numpy.newaxis,:]
    for kernel in kernels:
        def f(x):
            x = numpy.array((x,))[numpy.newaxis, :]
            return kde.evaluate.estimate_pdf_brute(x, training_points_1d, kernel=kernel)[0]
        integral = scipy.integrate.quad(f, -20, 20)[0]
    
        if numpy.isclose(integral, 1):
            print("    Test passed: 1D kernel '{:s}' integrates to {:f}".format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 1D kernel '{:s}' integrates to {:f}.".format(kernel, integral))
            print(terminalcolors.RESET, end='')
    
    # Test kernel normalization in one dimensional euclidean space, with multiple training points
    print("Testing kernel normalization in 1D Euclidean space, with multiple training points")
    training_points_1d = numpy.zeros(10)[:, numpy.newaxis]
    for kernel in kernels:
        def f(x):
            x = numpy.array((x,))[numpy.newaxis, :]
            return kde.evaluate.estimate_pdf_brute(x, training_points_1d, kernel=kernel)[0]
        integral = scipy.integrate.quad(f, -20, 20)[0]
    
        if numpy.isclose(integral, 1):
            print("    Test passed: 1D kernel '{:s}' integrates to {:f} with multiple "
                  "training points.".format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 1D kernel '{:s}' integrates to {:f} with multiple "
                  "training points.".format(kernel, integral))
            print(terminalcolors.RESET, end='')

    # Test kernel normalization in one dimensional n-Torus space
    print("Testing kernel normalization in 1D n-Torus space, with multiple training points")
    training_points_1d = 180*numpy.ones(1)[:, numpy.newaxis]
    for kernel in compactkernels:
        def f(x):
            x = numpy.array((x,))[numpy.newaxis, :]
            return kde.evaluate.estimate_pdf_brute(x, training_points_1d, kernel=kernel, metric='euclidean_distance_ntorus')[0]
        integral = scipy.integrate.quad(f, -180, 180)[0]
    
        if numpy.isclose(integral, 1):
            print("    Test passed: 1D kernel '{:s}' integrates to {:f}".format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 1D kernel '{:s}' integrates to {:f}.".format(kernel, integral))
            print(terminalcolors.RESET, end='')
    
    
    # Test kernel normalization in two dimensional euclidean space
    print("Testing kernel normalization in 2D Euclidean space")
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
    
        integral = scipy.integrate.dblquad(f, xmin, xmax, ymin, ymax, epsabs=.01)[0]
        if numpy.isclose(integral, 1, atol=0.01):
            print("    Test passed: 2D kernel '{:s}' integrates to {:f}".format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 2D kernel '{:s}' integrates to {:f}.".format(kernel, integral))
            print(terminalcolors.RESET, end='')

    # Test kernel normalization in two dimensional n-torus space
    print("Testing kernel normalization in 2-torus space")
    training_points_2d = 180*numpy.ones(2)[numpy.newaxis,:]
    for kernel in compactkernels:
        def ymin1(x):
            return -180
        def ymax1(x):
            return -170
        def ymin2(x):
            return 170
        def ymax2(x):
            return 180
        xmin1 = -180
        xmax1 = -170
        xmin2 = 170
        xmax2 = 180
    
        def f(x, y):
            x = numpy.array((x,y))[numpy.newaxis, :]
            return kde.evaluate.estimate_pdf_brute(x, training_points_2d, kernel=kernel, metric='euclidean_distance_ntorus')[0]
    
        integral1 = scipy.integrate.dblquad(f, xmin1, xmax1, ymin1, ymax1, epsabs=.01)[0]
        integral2 = scipy.integrate.dblquad(f, xmin1, xmax1, ymin2, ymax2, epsabs=.01)[0]
        integral3 = scipy.integrate.dblquad(f, xmin2, xmax2, ymin1, ymax1, epsabs=.01)[0]
        integral4 = scipy.integrate.dblquad(f, xmin2, xmax2, ymin2, ymax2, epsabs=.01)[0]

        integral = integral1 + integral2 + integral3 + integral4
        if numpy.isclose(integral, 1, atol=0.01):
            print("    Test passed: 2D kernel '{:s}' integrates to {:f}".format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 2D kernel '{:s}' integrates to {:f}.".format(kernel, integral))
            print(terminalcolors.RESET, end='')
    
    # Test kernel normalization in three dimensional euclidean space
    print("Testing kernel normalization in 3D Euclidean space")
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
        integral = scipy.integrate.nquad(f, [r for i in range(3)], opts={'epsabs': .1})[0]
        if numpy.isclose(integral, 1, atol=0.1):
            print("    Test passed: 3D kernel '{:s}' integrates to {:f}".format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 3D kernel '{:s}' integrates to {:f}.".format(kernel, integral))
            print(terminalcolors.RESET, end='')

if __name__ == "__main__":
    main()
