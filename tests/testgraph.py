#!/usr/bin/env python
import numpy
import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import kde
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec

class KDETest(object):
    '''
    Test the kde module.
    '''
    def __init__(self):
        matplotlib.rcParams['font.size'] = 6
        self.gridspec = gridspec.GridSpec(3,1)
        self.gridspec.update(hspace=0.2, wspace=0.2, left=0.2, right=0.95,
                             bottom=0.2, top = 0.95)
        self.fig = pyplot.gcf()
        self.fig.set_size_inches(8.8/2.54,8.8/2.54)
        self.ax1 = pyplot.subplot(self.gridspec[0,0])
        self.ax2 = pyplot.subplot(self.gridspec[1,0])
        self.ax3 = pyplot.subplot(self.gridspec[2,0])

        self.test_unweighted(self.ax1)
        self.test_gaussian(self.ax2)
        self.test_gamma(self.ax3)

        for ax in [self.ax1, self.ax2, self.ax3]:
            for kw in ['top', 'right']:
                ax.spines[kw].set_visible(False)
            ax.tick_params(direction='out', width=1)
            ax.set_xlim(0,15)
            ax.set_ylim(0,1)
        self.ax2.set_ylabel('probablity density')
        self.ax3.set_xlabel('x')
        pyplot.savefig('test.pdf')

    def test_unweighted(self, axis):
        data = numpy.array([1,1,2,2,3,3,3,4,6,8], dtype=float)
        #weights = numpy.array([5,8,9,4,5,2,3,1,1,2], dtype=float)
        #weights /= weights.sum()
        pdf = kde.KDE(data)
        xs = numpy.linspace(0,15,num=500)
        ys = pdf.evaluate(xs)
        axis.plot(xs, ys, color='black')

    def test_gaussian(self, axis):
        data = numpy.array([1,1,2,2,3,3,3,4,6,8], dtype=float)
        #weights = numpy.array([5,8,9,4,5,2,3,1,1,2], dtype=float)
        weights = numpy.array([1,1,1,1,1,1,1,1,1,1], dtype=float)
        weights /= weights.sum()
        pdf = kde.KDE(data, weights=weights)
        xs = numpy.linspace(0,15,num=500)
        ys = pdf.evaluate(xs)
        axis.plot(xs, ys, color='black')

    def test_gamma(self, axis):
        data = numpy.array([1,1,2,2,3,3,3,4,6,8], dtype=float)
        weights = numpy.array([5,8,9,4,5,2,3,1,1,2], dtype=float)
        weights /= weights.sum()
        pdf = kde.KDE(data, weights=weights, kernel='gamma')
        xs = numpy.linspace(0,15,num=500)
        ys = pdf.evaluate(xs)
        axis.plot(xs, ys, color='black')

class KDE2DTest(object):
    '''
    Test 2-dimensional kernel density estimation.
    '''
    def __init__(self):
        matplotlib.rcParams['font.size'] = 6
        fig = pyplot.gcf()
        ax = pyplot.gca()
        
        data = [(2,3),
                (2,3),
                (4,1),
                (7,1),
                (2,2),
                (5,4)]
        data = numpy.array(data)
        grid = numpy.linspace
        grid = numpy.mgrid[0:10:101j,0:10:101j].reshape(2,-1).T
        x = numpy.linspace(0,10,101)
        y = numpy.linspace(0,10,101)

        pdf = kde.KDE(data).evaluate(grid).reshape(101,101)[:,:].T
        ax.pcolormesh(x,y,pdf,cmap='magma')
        ax.plot(data[:,0], data[:,1], 'o', markersize=2, color='white')
        pyplot.savefig('2d.pdf')
        

if __name__ == "__main__":
    KDETest()
    pyplot.clf()
    KDE2DTest()
