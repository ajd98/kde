#!/usr/bin/env python
import sys
sys.path.append('../')
import numpy
import scipy.integrate
import kde

def main():
    training_data = [0]
    estimator = kde.KDE(training_data)
    query_points = numpy.linspace(-5,5,101)
    print(estimator.evaluate(query_points, cuda=True).shape)

if __name__ == "__main__":
    main()
