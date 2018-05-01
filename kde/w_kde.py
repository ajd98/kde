#!/usr/bin/env python
import argparse
import h5py
import importlib
import numpy
from .base import KDE
'''
Interface KDE with WESTPA simulation data. WKDE provides Python interface, and
WKDETool provides command line interface.

Run this module from the command lines as ``python w_kde.py --help`` for more
information on the command line interface.
'''

class WKDE(KDE):
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
            
class WKDETool(WKDE):
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
        self._eval_grid()
        self.go()

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

        parser.add_argument('--grid', default=None, required=True,
                            dest='gridstr', metavar='GRID_STRING',
                            help='Evaluate the kernel density estimate at each '
                                 'point in the specified grid. ``GRID_STRING`` '
                                 'should be string that Python can parse to a '
                                 'numpy array or list of numbers. While parsing'
                                 ' the string, numpy is made available as '
                                 '``numpy``.  For example, you may specify '
                                 '--grid "numpy.linspace(0,10,100)" in order to' 
                                 ' evaluate the kernel density estimate at 100 '
                                 'points evenly spaced between 0 and 10. The '
                                 'grid may also be specified via pure Python '
                                 'lists or list comprehensions, e.g., --grid '
                                 '"[0,1,2,3,4]" or --grid "[float(i)/10 for i '
                                 'in range(100)". For multi-dimensional grids, '
                                 'it is recommended to use the ``mgrid`` method'
                                 ' of numpy, e.g., --grid '
                                 '"numpy.mgrid[0:20:201j,0:10:101j].reshape(2,-1).T"'
                                 'to evaluate the kernel density estimate on a '
                                 'grid consisting of 201 points evenly spaced '
                                 'between 0 and 20 in the first dimension and '
                                 '101 points evenly spaced between 0 and 10 in '
                                 'the second dimension.'
                            )

        parser.add_argument('--output', default='pdf_estimate.dat', 
                            dest='output',
                            help='Save the result of evaluating the kernel '
                                 'density estimate at each point in '
                                 '``GRID_STRING`` in the file ``OUTPUT``. '
                                 'Values are saved in ASCII format, with (if '
                                 'applicable) the rows indexing the data point,'
                                 ' and the columns indexing the dimension.'
                            )
        parser.add_argument('--cuda', action='store_true', dest=cuda,
                            help="If this flag is provided, use the CUDA "
                                 "backend for evaluation of the kernel density "
                                 "estimate."
                            )


        self.args = parser.parse_args() 
        self.h = self.args.bw

    def _eval_grid(self):
        self.grid = eval(self.args.gridstr)

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

    def go(self):
        result = self.evaluate(self.grid, cuda=self.args.cuda)
        numpy.savetxt(self.args.output, result)

if __name__ == "__main__":
    kdestimator = WKDETool()
