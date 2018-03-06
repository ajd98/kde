#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('../')
sys.path.append('../kde')
import kde.evaluate
import numpy
from warningcolors import terminalcolors
from normalization import kernels
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec

print("Testing kernel shapes")

gs = gridspec.GridSpec(len(kernels),1, hspace=0.3)
fig = pyplot.gcf()
fig.set_size_inches(3.92,8)


training_points_1d = numpy.zeros(1)[numpy.newaxis,:]
query_points = numpy.linspace(-5,5,201)[:,numpy.newaxis]
for ikernel, kernel in enumerate(kernels):
    ax = fig.add_subplot(gs[ikernel,0])
    ys = kde.evaluate.estimate_pdf_brute(query_points, training_points_1d, kernel=kernel)
    ax.plot(query_points, ys)
    ax.set_xlim(query_points.min(), query_points.max())
    ax.set_ylim(0,1)
    ax.text(0.01, 0.9, kernel, transform=ax.transAxes)

pyplot.savefig('kernelshapes.pdf')

