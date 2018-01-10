#!/usr/bin/env python
import numpy
import scipy.special
import statistics

class KDE(object):
    def __init__(self, data, kernel='gaussian', weights = None, bw=1):
        self.data = data
        self.set_kernel_type(kernel)
        self.h = bw
        self.N = data.shape[0]
        if weights is not None:
            self.weights = weights
        else:
            self.weights = numpy.ones(self.data.shape[0], dtype=float)\
                           /float(self.data.shape[0])

    def _ndarray_is_1d(self, arr):
        '''
        Returns true if ``arr`` is one-dimensional, or if all axes except the 
        zeroth are length 1.
        '''
        if arr.ndim == 1:
            return True
        else:
            for dim in xrange(1,arr.ndim):
                if arr.shape[dim] > 1:
                    return False
        return True

    def set_kernel_type(self, kernel):
        # Check to make sure the kernel is compatible with the data shape
        if kernel == 'gamma':
            if not self._ndarray_is_1d(self.data):
                raise Exception('Gamma kernel is only compatible with '
                                '1-dimensional data, but data is shape {}'\
                                .format(self.data.shape))
        self.kernel_type = kernel

    def evaluate(self, points):
        if self.kernel_type == 'gaussian':
            return self._compute_density_gaussian(points)
        elif self.kernel_type == 'gamma':
            return self._compute_density_gamma(points)
        else:
            return 'kernel type {} not implemented'.format(self.kernel_type)

    def sum_except_along(self, arr, except_this_axis=0):
        '''
        Sum the array along all axes except the specified axis index

        arr: numpy.ndarray
        except_this_axis: int, denotes axis not to sum along

        Needs to be tested.
        '''
        result = arr
        offset = 0
        for iax in xrange(arr.ndim):

            # Keep collapsing the zeroth axis until we reach 
            # ``except_this_axis``
            if iax < except_this_axis:
                result = result.sum(axis=0)
            elif iax == except_this_axis:
                pass
            # After passing ``except_this_axis``, keep collapsing along the
            # first axis until we run out of axes. The zeroth axis now
            # corresponds to ``except_this_axis``.
            else: #if iax> except_this axis
                result = result.sum(axis=1)
        return result


    def _compute_density_gaussian(self, points):
        '''
        Compute the kernel density estimate of the probability density
        for each point in the vector ``points``.
        '''
        result = numpy.empty(points.shape, dtype = numpy.float64)
        for idx, p in enumerate(points):
            difference = self.data - p
            scaled_diff = difference/self.h**2
            if self.data.ndim > 1 :
                var = numpy.divide(self.sum_except_along(numpy.multiply(scaled_diff, difference), 0),-2)
            else:
                var = numpy.divide(numpy.multiply(scaled_diff, difference),-2)
            vals = numpy.exp(var)
            result[idx] = numpy.multiply(vals, self.weights).sum(axis=0)
        h = self.h
        result /= (self.weights.sum()*numpy.sqrt(2*numpy.pi*h**2))
        return result

    def _compute_density_gaussian_experimental(self, points):
        '''
        Compute the kernel density estimate of the probability density
        for each point in the vector ``points``.
        '''
        result = numpy.empty(points.shape, dtype = numpy.float64)
        self._compute_inv_cov()
        for idx, p in enumerate(points):
            difference = self.data - p

            #????
            #scaled_diff = numpy.dot(self.inv_cov[0][0], difference)
            scaled_diff = difference/self.h**2
            #scaled_diff = numpy.dot(difference, self.inv_cov)

            # Commented out for testing...
            #var = numpy.divide(numpy.sum(numpy.multiply(scaled_diff, difference), axis=0),-2)

            var = numpy.divide(numpy.sum(numpy.multiply(scaled_diff, difference), axis=1),-2)
            vals = numpy.exp(var)
            result[idx] = numpy.multiply(vals, self.weights).sum(axis=0)
        # Testing normalization...
        h = self.h
        result /= (self.weights.sum()*numpy.sqrt(2*numpy.pi*h**2))
        return result

    def _altered_gamma(self, p, points):
        '''
        Evalute at ``points`` the gamma distribution with mode ``p`` and
        standard deviation self.h.

        ``p``: (int) mode of the distribution 
        ``points``: (numpy.ndarray) Points at which to evaluate 
        '''
        h = self.h
        k = (numpy.sqrt(4*h**2*p**2+p**4)+2*h**2+p**2)/(2*h**2)
        theta = h/numpy.sqrt(k)

        return points**(k-1)*numpy.exp(-points/theta)/(theta**k*scipy.special.gamma(k)) 

    def _altered_gamma_2(self, p, points):
        '''
        Evalute at p the sum of the gamma distributions with modes at 
        ``points``. The scale and shape parameters of the distributions are
        varied such that the modes are at ``points`` and the standard deviation
        (square root of variance) is self.h.

        ``p``: (int) particular point at which to evaluate
        ``points``: (numpy.ndarray) Modes of the gamma distributions
        '''
        h = self.h
        # A vector, which is the k for every datapoint
        k = (numpy.sqrt(4*h**2*points**2+points**4)+2*h**2+points**2)/(2*h**2)
        # A vector, which is the theta for every datapoint
        theta = h/numpy.sqrt(k)
        return numpy.multiply(p**(k-1)*numpy.exp(-p/theta)/(theta**k*scipy.special.gamma(k)), self.weights).sum()

    def _compute_density_gamma(self, points):
        '''
        Compute the kernel density estimate of the probability density
        at each point in the vector ``points``, using a gamma kernel.

        pdf of gamma is f(x) = 1/[Gamma(k)*theta^k]*x^(k-1)*exp(-x/theta)
        mode is (k-1)*theta (for k>1)
        variance is k*theta^2

        We choose k, theta such that (k-1)*theta is at the particular point, and 
        sqrt(variance) = sqrt(k)*theta is bw

        then theta= h/sqrt(k)
        (k-1)*h/sqrt(k) = m
        => k = [sqrt(4*h^2*m^2+m^4) + 2*h^2+m^2]/[2*h^2]
        theta = h/sqrt(k)
        '''
        result = numpy.empty(points.shape, dtype = numpy.float64)
        for idx, p in enumerate(points):
            result[idx] = self._altered_gamma_2(p, self.data)
        result /= self.weights.sum()
        return result

    def _compute_inv_cov(self):
        cov_mat = statistics.weightedcovariance(self.data, self.weights)
        self.inv_cov = numpy.linalg.inv(cov_mat)
        self.inv_cov = self.inv_cov/ self.h**2

