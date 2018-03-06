#!/usr/bin/env python
import numpy
import scipy.special
import statistics
import kde.evaluate as evaluate

class KDE(object):
    kernels = ['bump', 'cosine', 'epanechnikov', 'gaussian', 'logistic',
               'quartic', 'tophat', 'triangle', 'tricube']

    def __init__(self, training_points, kernel='gaussian', weights = None, bw=1):
        self.training_points = numpy.asarray(training_points)

        # Make training points two dimensional.
        if self.training_points.ndim > 2 or self.training_points.ndim == 0:
            raise ValueError("Training points can only be 1 or 2 dimensional.")
        elif self.training_points.ndim == 1:
            self.training_points = self.training_points[:, numpy.newaxis]
        elif self.training_points.ndim == 2:
            pass

        self.set_kernel_type(kernel)
        self.bw = bw

        # Normalize the weights, making a copy of the array if necessary
        if weights is not None:
            self.weights = numpy.require(weights, dtype=numpy.float64)
            s = self.weights.sum()
            if s != 1:
                self.weights = numpy.copy(weights)/s 
        else:
            self.weights = None

    def set_kernel_type(self, kernel):
        if not kernel in self.kernels:
            raise ValueError("Invalid kernel {:s}. Valid kernels include: \n{:s}"\
                             .format(kernel, repr(self.kernels)))
        self.kernel_type = kernel

    def evaluate(self, points):
        '''
        Evaluate the kernel density estimate at each point in ``points``
        '''
        # Make sure points is the correct shape
        points = numpy.require(points, dtype=numpy.float64)
        if points.ndim > 2 or points.ndim == 0:
            raise ValueError("Dimension of ``points`` must be the same as the "
                             "training_points.")
        elif points.ndim == 1:
            points_ = points[:,numpy.newaxis]
        elif points.ndim == 2:
            points_ = points 

        if points.shape[1] != self.training_points.shape[1]:
            raise ValueError("``points`` has {:d} features while "
                    "``training_points`` has {:d} features. Number of features "
                    "must be the same.".format(points.shape[1],
                                               self.training_points.shape[1]))

        result = evaluate.estimate_pdf_brute_force(points_, self.training_points,
                                                   bandwidth=self.bw, 
                                                   weights=self.weights,
                                                   metric=self.metric,
                                                   kernel=self.kernel)

        # Return array of same shape as input ``points``
        return result.reshape(points.shape)
