#!/usr/bin/env python
from __future__ import print_function
import argparse
import sys
sys.path.append('../')
sys.path.append('../kde')
import kde.evaluate
import numpy
from normalization import kernels
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


def test1d(cuda=False):
    print("Testing kernel shapes")
    
    gs = gridspec.GridSpec(len(kernels),1, hspace=0.5)
    fig = pyplot.gcf()
    fig.set_size_inches(3.92,8)
    matplotlib.rcParams['font.size'] = 7
    
    training_points_1d = numpy.zeros(1)[numpy.newaxis,:]
    query_points = numpy.linspace(-5,5,201)[:,numpy.newaxis]
    estimator = kde.KDE(training_points_1d)
    for ikernel, kernel in enumerate(kernels):
        ax = fig.add_subplot(gs[ikernel,0])
        estimator.set_kernel_type(kernel)
        ys = estimator.evaluate(query_points, cuda=True)
        ax.plot(query_points, ys)
        ax.set_xlim(query_points.min(), query_points.max())
        ax.set_ylim(0,1)
        ax.text(0.01, 0.7, kernel, transform=ax.transAxes)
    
    pyplot.savefig('1dkernels.pdf')
    pyplot.clf()

def test2d(cuda=False):
    fig = pyplot.gcf()
    fig.set_size_inches(3.92,8)
    training_points_2d = numpy.zeros(2)[numpy.newaxis,:]
    X = numpy.linspace(-2,2,101)[:,numpy.newaxis]
    Y = numpy.linspace(-2,2,101)[:,numpy.newaxis]
    X, Y = numpy.meshgrid(X, Y)
    query_pts = numpy.vstack((X.ravel(), Y.ravel())).T
    
    estimator = kde.KDE(training_points_2d)
    gs = gridspec.GridSpec(int(numpy.ceil(len(kernels)/2.)),2, hspace=0.5)
    for ikernel, kernel in enumerate(kernels):
        ax = fig.add_subplot(gs[ikernel//2,ikernel%2], projection='3d')
        estimator.set_kernel_type(kernel)
        Z = estimator.evaluate(query_pts, cuda=True)
        ax.plot_surface(X, Y, Z.reshape(X.shape[0], -1).T)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(0,1)
        ax.set_xlabel(kernel)
    pyplot.savefig('2dkernels.pdf')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', dest='cuda', action='store_true',
                        help="Run tests using cuda backend.")
    args = parser.parse_args()
    test1d(cuda=args.cuda)
    test2d(cuda=args.cuda)
    
    print("See 1dkernels.pdf and 2dkernels.pdf for output.")

if __name__ == "__main__":
    main()
