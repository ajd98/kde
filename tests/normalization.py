#!/usr/bin/env python
from __future__ import print_function
import argparse
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
def test_1d_euclidean(cuda=False):
    # Test kernel normalization in one dimensional euclidean space
    print("Testing kernel normalization in 1D Euclidean space")
    training_points_1d = numpy.zeros(1)[numpy.newaxis,:]
    estimator = kde.KDE(training_points_1d, bw=1)
    for kernel in kernels:
        estimator.set_kernel_type(kernel)
        def f(x):
            x = numpy.array((x,))[numpy.newaxis, :]
            return estimator.evaluate(x, cuda=cuda)[0]
        integral = scipy.integrate.quad(f, -20, 20)[0]
    
        if numpy.isclose(integral, 1):
            print("    Test passed: 1D kernel '{:s}' integrates to {:.02f}"\
                  .format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 1D kernel '{:s}' integrates to {:.02f}."\
                  .format(kernel, integral))
            print(terminalcolors.RESET, end='')

def test_1d_euclidean_bw2(cuda=False):
    # Test kernel normalization in one dimensional euclidean space
    print("Testing kernel normalization in 1D Euclidean space, with bandwidth "
          "of 2")
    training_points_1d = numpy.zeros(1)[numpy.newaxis,:]
    estimator = kde.KDE(training_points_1d, bw=2)
    for kernel in kernels:
        estimator.set_kernel_type(kernel)
        def f(x):
            x = numpy.array((x,))[numpy.newaxis, :]
            return estimator.evaluate(x, cuda=cuda)[0]
        integral = scipy.integrate.quad(f, -20, 20)[0]
    
        if numpy.isclose(integral, 1, atol=0.001):
            print("    Test passed: 1D kernel '{:s}' integrates to {:.02f}"\
                  .format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 1D kernel '{:s}' integrates to {:.02f}."\
                  .format(kernel, integral))
            print(terminalcolors.RESET, end='')

def test_1d_euclidean_multiple_training(cuda=False):
    # Test kernel normalization in one dimensional euclidean space, with 
    # multiple training points
    print("Testing kernel normalization in 1D Euclidean space, with multiple "
          " training points")
    training_points_1d = numpy.zeros(10)[:, numpy.newaxis]
    estimator = kde.KDE(training_points_1d, bw=1)
    for kernel in kernels:
        estimator.set_kernel_type(kernel)
        def f(x):
            x = numpy.array((x,))[numpy.newaxis, :]
            return estimator.evaluate(x, cuda=cuda)[0]
        integral = scipy.integrate.quad(f, -20, 20)[0]
    
        if numpy.isclose(integral, 1):
            print("    Test passed: 1D kernel '{:s}' integrates to {:.02f} with multiple "
                  "training points.".format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 1D kernel '{:s}' integrates to {:.02f} with multiple "
                  "training points.".format(kernel, integral))
            print(terminalcolors.RESET, end='')

def test_1d_euclidean_complex(cuda=False):
    # Test kernel normalization in one dimensional euclidean space
    print("Testing kernel normalization in 1D Euclidean space with alternate "
          "training set.")
    training_points_1d = numpy.array((0,0,1,2,5,-4))
    estimator = kde.KDE(training_points_1d, bw=1)
    for kernel in kernels:
        estimator.set_kernel_type(kernel)
        def f(x):
            x = numpy.array((x,))[numpy.newaxis, :]
            return estimator.evaluate(x, cuda=cuda)[0]
        integral = scipy.integrate.quad(f, -20, 20)[0]
    
        if numpy.isclose(integral, 1):
            print("    Test passed: 1D kernel '{:s}' integrates to {:.02f}"\
                  .format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 1D kernel '{:s}' integrates to {:.02f}."\
                  .format(kernel, integral))
            print(terminalcolors.RESET, end='')

def test_1d_euclidean_ntorus(cuda=False):
    # Test kernel normalization in one dimensional n-Torus space
    print("Testing kernel normalization in 1D n-Torus space, with multiple "
          "training points")
    training_points_1d = 180*numpy.ones(1)[:, numpy.newaxis]
    estimator = kde.KDE(training_points_1d, bw=1, 
                        metric='euclidean_distance_ntorus')
    for kernel in compactkernels:
        estimator.set_kernel_type(kernel)
        def f(x):
            x = numpy.array((x,))[numpy.newaxis, :]
            return estimator.evaluate(x, cuda=cuda)[0]
        integral = scipy.integrate.quad(f, -180, 180)[0]
    
        if numpy.isclose(integral, 1):
            print("    Test passed: 1D kernel '{:s}' integrates to {:.02f}"\
                  .format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 1D kernel '{:s}' integrates to {:.02f}."\
                  .format(kernel, integral))
            print(terminalcolors.RESET, end='')

def test_2d_euclidean(cuda=False):
    # Test kernel normalization in two dimensional euclidean space
    print("Testing kernel normalization in 2D Euclidean space")
    training_points_2d = numpy.zeros(2)[numpy.newaxis,:]
    estimator = kde.KDE(training_points_2d, bw=1)
    for kernel in kernels:
        estimator.set_kernel_type(kernel)
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
            return estimator.evaluate(x, cuda=cuda)[0]
    
        integral = scipy.integrate.dblquad(f, xmin, xmax, ymin, ymax, epsabs=.01)[0]
        if numpy.isclose(integral, 1, atol=0.01):
            print("    Test passed: 2D kernel '{:s}' integrates to {:.02f}"\
                  .format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 2D kernel '{:s}' integrates to {:.02f}."\
                  .format(kernel, integral))
            print(terminalcolors.RESET, end='')

def test_2d_euclidean_ntorus(cuda=False):
    # Test kernel normalization in two dimensional n-torus space
    print("Testing kernel normalization in 2-torus space")
    training_points_2d = 180*numpy.ones(2)[numpy.newaxis,:]
    estimator = kde.KDE(training_points_2d, bw=1, metric='euclidean_distance_ntorus')
    for kernel in compactkernels:
        estimator.set_kernel_type(kernel)
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
            return estimator.evaluate(x, cuda=cuda)[0]
    
        integral1 = scipy.integrate.dblquad(f, xmin1, xmax1, ymin1, ymax1, epsabs=.01)[0]
        integral2 = scipy.integrate.dblquad(f, xmin1, xmax1, ymin2, ymax2, epsabs=.01)[0]
        integral3 = scipy.integrate.dblquad(f, xmin2, xmax2, ymin1, ymax1, epsabs=.01)[0]
        integral4 = scipy.integrate.dblquad(f, xmin2, xmax2, ymin2, ymax2, epsabs=.01)[0]

        integral = integral1 + integral2 + integral3 + integral4
        if numpy.isclose(integral, 1, atol=0.01):
            print("    Test passed: 2D kernel '{:s}' integrates to {:.02f}"\
                  .format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 2D kernel '{:s}' integrates to {:.02f}."\
                  .format(kernel, integral))
            print(terminalcolors.RESET, end='')

def test_3d_euclidean(cuda=False):
    # Test kernel normalization in three dimensional euclidean space
    print("Testing kernel normalization in 3D Euclidean space")
    training_points_3d = numpy.zeros(3)[numpy.newaxis,:]
    estimator = kde.KDE(training_points_3d, bw=1)
    for kernel in kernels:
        estimator.set_kernel_type(kernel)
        if kernel in ['logistic', 'gaussian']:
            xmin = -10
            xmax = 10
    
        else:
            xmin = -1
            xmax = 1
    
        def f(x, y, z):
            x = numpy.array((x,y,z))[numpy.newaxis, :]
            return estimator.evaluate(x, cuda=cuda)[0]
    
        r = (xmin, xmax)
        integral = scipy.integrate.nquad(f, [r for i in range(3)], opts={'epsabs': .1})[0]
        if numpy.isclose(integral, 1, atol=0.1):
            print("    Test passed: 3D kernel '{:s}' integrates to {:.02f}"\
                  .format(kernel, integral))
        else:
            print(terminalcolors.FAIL, end='')
            print("    TEST FAILED: 3D kernel '{:s}' integrates to {:.02f}."\
                  .format(kernel, integral))
            print(terminalcolors.RESET, end='')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', dest='cuda', action='store_true',
                        help="Run tests using cuda backend.")
    args = parser.parse_args()
    test_1d_euclidean(cuda=args.cuda)
    test_1d_euclidean_bw2(cuda=args.cuda)
    test_1d_euclidean_multiple_training(cuda=args.cuda)
    test_1d_euclidean_complex(cuda=args.cuda)
    test_1d_euclidean_ntorus(cuda=args.cuda)
    test_2d_euclidean(cuda=args.cuda)
    test_2d_euclidean_ntorus(cuda=args.cuda)
    test_3d_euclidean(cuda=args.cuda)

if __name__ == "__main__":
    main()
