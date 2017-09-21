#!/usr/bin/env python
'''
Classes for building kernel density estimates from WESTPA simulation data.
Written by Alex DeGrave
'''
import argparse
import h5py
import importlib
import numpy
import pylab
import scipy
import scipy.special


def weightedcovariance(m, w):
    '''
    Returns an unbiased estimate of the weighted covariance matrix for the 
    N by k matrix ``m`` where N is the number of samples and k is the number
    of dimensions of data per sample, and the corresponding length-N vector
    ``w`` representing the weight of each sample.

    >>> mat = numpy.array([[1,5,7],
                           [3,4,1],
                           [2,7,5],
                           [5,4,3]])
    >>> w = numpy.array([.2,.4,.1,.3])
    >>> weightedcovariance(mat, w)
        
    array([[ 2.98571429, -1.07142857, -2.6       ],
           [-1.07142857,  1.21428571,  1.85714286],
           [-2.6       ,  1.85714286,  7.37142857]])

    '''
    # Normalize weights
    w /= w.sum()
    # Weight each observation by its respective weight.
    wm = numpy.multiply(w,m.T).T 

    #Calculate the mean of each column
    avg = wm.sum(axis=0) 
    #print(avg)
    submat = m - avg 
    wsum = w.sum()
    inv_bias = wsum/(numpy.square(wsum) - w.dot(w))
    cov = inv_bias*submat.T.dot(numpy.multiply(w, submat.T).T) 
    return cov

def wvariance(m, w):
    '''
    Return the weighted variance for the vectory of samples ``m`` with 
    associated weights ``w``.
    '''
    wsum = w.sum()
    mean = (w*m).sum()/wsum 
    #print(mean)
    inv_bias = wsum/(numpy.square(wsum) - w.T.dot(w))
    cov = inv_bias * numpy.sum(w*numpy.square((m - mean)))
    return cov

class KDE:
    def __init__(self, data, kernel='gaussian', weights = None, bw=1):
        self.data = data
        self.kernel_type = kernel
        self.h = bw
        self.N = data.shape[0]
        if weights is not None:
            self.weights = weights
        else:
            self.weights = numpy.ones(self.data.shape[0], dtype=float)\
                           /float(self.data.shape[0])

    def set_kernel_type(self, kernel):
        self.kernel_type = kernel

    def evaluate(self, points):
        if self.kernel_type == 'gaussian':
            return self._compute_density_gaussian(points)
        elif self.kernel_type == 'gamma':
            return self._compute_density_gamma(points)
        else:
            return 'kernel type {} not implemented'.format(self.kernel_type)

    def _sum_except_along(self, arr, except_this_axis=0):
        '''
        Sum the array along all axes except the specified axis index

        arr: numpy.ndarray
        except_this_axis: int, denotes axis not to sum along

        Needs to be tested.
        '''
        result = arr
        offset = 0
        for iax in xrange(arr.ndims):

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
        cov_mat = weightedcovariance(self.data, self.weights)
        self.inv_cov = numpy.linalg.inv(cov_mat)
        self.inv_cov = self.inv_cov/ self.h**2
        print(self.inv_cov)

class wKDE(KDE):
    '''
    Wrapper class for KDE, providing a way to interface with data from WESTPA
    simulations.
    '''
    def __init__(self, westh5, first_iter=None, last_iter=None, load_func=None, bw=1):
        '''
        Initialize the KDE class. Load data into memory and format, and 
        set up various parameters.
        '''

        # Make the main west.h5 file accessible as an attribute
        self.westh5 = westh5

        # If the user does not specify a range of iterations to use, parse the 
        # west.h5 to get the entire range of iterations. 
        self.first_iter = first_iter
        self.last_iter = last_iter
        if self.first_iter is None or self.last_iter is None:
            self._get_iter_range()
        # By default, use the progress coordinate for data.
        if load_func is None:
            self._load_func = self._get_pcoord
        else:
            self._load_func = load_func

        self._scan_data()
        self._load_data()
        self.set_kernel_type('gaussian')
        self.h = bw


    def _raise_error(self, err, errmsg):
        '''
        Exit with error ``err``, first printing ``errmsg``.
        '''
        print(errmsg)
        raise err

    def _get_pcoord(self, iter_group, n_iter):
        return iter_group['pcoord']

    def _get_iter_range(self):
        '''
        Get the range of iterations from which to pull data, and store as a 
        tuple (first_iter, last_iter) in self.iter_range. Check to make sure
        the user-specified iterations (if applicable) are in the HDF5 file.
        '''
        if self.first_iter is None:
            self.first_iter = 1

        if self.last_iter is None:
            # Get the last iteration in the HDF5 file.
            keys = self.westh5['iterations'].keys()
            skeys = sorted(keys)
            self.last_iter = int(skeys[-1][5:13])

        self.iter_range = (self.first_iter, self.last_iter)

    def _scan_data(self):
        '''    
        Scan over the iterations and build empty arrays of a suitable size to
        hold all the data.
        '''    
        N_sum = 0
        k = 0
        for iter_idx in range(self.iter_range[0], self.iter_range[1] + 1):
            iter_group = self.westh5['iterations/iter_%08d'%iter_idx]
            iter_data = self._load_func(iter_group, iter_idx)
            shape = iter_data.shape
            N_sum += shape[0]*shape[1] 
            k = shape[2]
        self.data = numpy.empty((N_sum, k), dtype=numpy.float64)

        self.weights = numpy.empty((N_sum), dtype=numpy.float64)
        
    def _load_data(self): 
        '''
        Load data from the specified west.h5 file into memory as two arrays.
        The array ``self._data`` will be an N by k array containing k 
        coordinate values for each of N samples.  The array ``self._weights``
        will be a length-N vector containing the weight corresponding to each 
        sample in ``self._data``. 
        '''
        arr_idx = 0
        # The dimensionality of the dataset
        k = self.data.shape[1]

        for iter_idx in range(self.iter_range[0], self.iter_range[1] + 1):
            iter_group = self.westh5['iterations/iter_%08d'%iter_idx]
            iter_data = self._load_func(iter_group, iter_idx)
            # Reshape the data by collapsing along the timepoint index. Treat
            # each timepoint as a separate observation.
            m = iter_data.shape[1]
            iter_data = numpy.array(iter_data).reshape(-1, k) 
            # The total number of datapoints in this iteration.
            # In other words, n_points = number of segments * timepoints per iteration 
            n_points = iter_data.shape[0]
            # Stick the data into this array.  Memory efficiency is important 
            # in self.data, as it could be huge.
            self.data[arr_idx: arr_idx + n_points] = iter_data
            # Load the weights for this iteration
            iter_weights = self.westh5['iterations/iter_%08d/seg_index'%iter_idx]['weight']
            iter_weights = numpy.array(iter_weights)
            iter_weights = numpy.repeat(iter_weights, m)
            self.weights[arr_idx: arr_idx + n_points] = iter_weights 

            # This makes sure data goes in every slot of self.data without
            # overwriting other data.
            arr_idx += n_points

    def go(self, points):
        self.set_kernel_type('gaussian')
        return self.evaluate(points)
            
class wKDETool(wKDE):
    '''
    Wrapper class for wKDE, providing a command line interface.
    '''
    def __init__(self):
        self._parse_args()
        self._import_func()
        self._load_westh5()
        self._get_iter_range() 
        self.set_kernel_type('gaussian')
        self._scan_data()
        self._load_data()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-W', default='./west.h5',
                            dest='westh5_path', 
                            help='Load data from WESTH5_PATH.',
                            type=str)

        parser.add_argument('--first-iter', default=None,
                            dest='first_iter',
                            help='Plot data starting at iteration FIRST_ITER. '
                                 'By default, plot data starting at the first '
                                 'iteration. ',
                            type=int)

        parser.add_argument('--last-iter', default=None,
                            dest='last_iter',
                            help='Plot data up to and including iteration '
                                 'LAST_ITER. By default, plot data up to and '
                                 'including the last iteration in the specified '
                                 'west.h5 file.  Use with ``--pdist-input``.',
                            type=int)

        parser.add_argument('--construct-func', default=None,
                            dest='load_func',
                            help='Load data for each iteration by calling the ' 
                                 'python function ``LOAD_FUNC`` for each '
                                 'iteration in the specified range. Specify '
                                 '``LOAD_FUNC`` as module.function, where '
                                 '``module`` is a Python file and ``function`` '
                                 ' is a function in the file. Call the '
                                 'function as ``load_func(iter_group, n_iter)``'
                                 ' where ``iter_group`` is the HDF5 group from '
                                 ' the west.h5 file for an iteration, and '
                                 'n_iter is the index of the iteration. The '
                                 'function should return an N*m*k array, where '
                                 'N is the number of segments, m is the number '
                                 'of timepoints per iteration, and k is the '
                                 'dimensionality of the data for which to '
                                 'estimate the probability density function.'
                                 'By default, use all dimensions of the '
                                 'progress coordinate.',
                            type=str)
        parser.add_argument('--bw', default=1,
                            dest='bw',
                            help='For gaussian kernels, use ``bw`` as sigma.',
                            type=float)

        self.args = parser.parse_args() 
        self.h = self.args.bw

    def _import_func(self):
        '''
        Non-public.
        Load the user-specified dataset construction function.
        '''
        if self.args.load_func is not None:
            # Parse the user-specified string for the mudule and class/function name.
            module_name, attr_name = self.args.load_func.split('.',1)
            # import the module ``module_name`` and make the function/class 
            # accessible as ``self._load_func``.
            self._load_func = getattr(importlib.import_module(module_name), attr_name)
        else:
            self._load_func = self._get_pcoord

    def _load_westh5(self):
        '''
        Non-public.
        Create the attribute self.westh5, a h5py File handle for the
        user-specified WESTPA data file (usually west.h5)
        '''
        try: 
            self.westh5 = h5py.File(self.args.westh5_path,'r') 
        except IOError:
            self._raise_error(IOError, "Error reading specified HDF5 file.")

    def _get_iter_range(self):
        '''
        Get the range of iterations from which to pull data, and store as a 
        tuple (first_iter, last_iter) in self.iter_range. Check to make sure
        the user-specified iterations (if applicable) are in the HDF5 file.
        '''
        if self.args.first_iter is not None:
            first_iter = self.args.first_iter
            try:
                self.westh5['iterations/iter_%08d/'%first_iter]
            except KeyError:
                self._raise_error(KeyError, 'Error loading the specified '
                                            'iteration %d.' % first_iter )
        else:
            first_iter = 1

        if self.args.last_iter is not None:
            last_iter = self.args.last_iter
            try:
                self.westh5['iterations/iter_%08d/'%last_iter]
            except KeyError:
                self._raise_error(KeyError, 'Error loading the specified '
                                            'iteration %d.' % last_iter )
        else:
            # Get the last iteration in the HDF5 file.
            keys = self.westh5['iterations'].keys()
            skeys = sorted(keys)
            last_iter = int(skeys[-1][5:13])

        self.iter_range = (first_iter, last_iter)

###############################################################################
if __name__ == "__main__":
    kdestimator = wKDETool()
    p = numpy.linspace(0,30,200)
    density = kdestimator.evaluate(p)
    #print(density)
    ys = -1*numpy.log(density)
    ys -= numpy.nanmin(ys)
    pylab.plot(p, ys)
    pylab.legend()
    pylab.show()

