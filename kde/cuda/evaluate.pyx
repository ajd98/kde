# cython: boundscheck=False
#
import numpy
import kde.evaluate.kernel_coefficients as kernel_coefficients
from libc.stdlib cimport malloc, free

COMPACT_KERNELS=['bump', 'cosine', 'epanechnikov', 'quartic', 'tophat', 
                 'triangle', 'tricube']

cdef extern from "_evaluate.h":
    void cuda_evaluate(const double* query_points,
                       const double* training_points, 
                       const double* weights,
                       char* metric_s,
                       char* kernel_s,
                       double h, 
                       double* result,
                       int nquery, int ntrain, int ndim) nogil

def estimate_pdf_brute(query_points, training_points, bandwidth=1, weights=None,
                       metric='euclidean_distance', kernel='gaussian'):
    '''
    estimate_pdf_brute(query_points, training_points, 
                       metric='euclidean_distance', kernel='gaussian')


    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(kernel_func(metric_func(x-xi)/h))

    ----------
    Parameters
    ----------
    query_points: (numpy.ndarray)
    training_points: (numpy.ndarray)
    bandwidth: (float) The bandwidth of the kernel; ``h`` in the equation above.
    weights: (None or float) Weights for each training point.  If ``None``, 
      training points are uniformly weighted.
    metric: (str) options are 'euclidean_distance' and 'euclidean_distance_ntorus'
    kernel: (str) options are:
     - 'bump'
     - 'cosine'
     - 'epanechnikov'
     - 'gaussian'
     - 'logistic'
     - 'quartic'
     - 'tophat'
     - 'triangle'
     - 'tricube'

     -------
     Returns
     -------
     result: (numpy.ndarray) The value of the kernel density estimate at each 
       point in ``query_points``

    '''
    if training_points.shape[1] != query_points.shape[1]:
        raise TypeError("Training points and query points must have same "
                        "dimension but have dimension {:d} and {:d}"\
                        .format(training_points.shape[1], query_points.shape[1]))

    cdef int nquery = query_points.shape[0]
    cdef int ntrain = training_points.shape[0]
    cdef int ndim = training_points.shape[1]

    coeffdict = {'bump': kernel_coefficients.bump_coefficient,
                 'cosine': kernel_coefficients.cosine_coefficient,
                 'epanechnikov': kernel_coefficients.epanechnikov_coefficient,
                 'gaussian': kernel_coefficients.gaussian_coefficient,
                 'logistic': kernel_coefficients.logistic_coefficient,
                 'quartic': kernel_coefficients.quartic_coefficient,
                 'tophat': kernel_coefficients.tophat_coefficient,
                 'triangle': kernel_coefficients.triangle_coefficient,
                 'tricube': kernel_coefficients.tricube_coefficient}

    # Parse keyword arguments
    try:
        coeff = coeffdict[kernel](ndim)
    except KeyError:
        raise KeyError("Kernel {:s} not found.".format(kernel))

    if metric == 'euclidean_distance_ntorus':
       if not kernel in COMPACT_KERNELS:
           raise ValueError("Kernel {:s} is not compact, and therefore invalid "
                            "for use with n-torus space.".format(kernel))
       elif bandwidth > 180:
           raise ValueError("Bandwidth {:f} is too large for use with n-torus "
                            "space. Bandwidth must be less than or equal to 180."\
                            .format(bandwidth))

    if weights is None:
        weights = numpy.ones(ntrain, dtype=numpy.float64)/ntrain

    ###### Wrapper for CUDA ######
    #if cuda, initialize c-style arrays. The double-underscore arrays denote
    # c-style buffers
    cdef double* __query_points = <double*>malloc(nquery*ndim*sizeof(double))
    if __query_points == NULL:
        raise MemoryError("Failed to allocate __query_points")
    cdef double* __training_points = <double*>malloc(ntrain*ndim*sizeof(double))
    if __training_points == NULL:
        raise MemoryError("Failed to allocate __training_points")
    cdef double* __weights = <double*>malloc(ntrain*sizeof(double))
    if __weights == NULL:
        raise MemoryError("Failed to allocate __weights")
    cdef double* __result = <double*>malloc(nquery*sizeof(double))
    if __result == NULL:
        raise MemoryError("Failed to allocate __result")
    cdef char* __metric = metric
    cdef char* __kernel = kernel

    # assign values to arrays
    for i in range(nquery):
        for j in range(ndim):
            __query_points[ndim*i+j] = query_points[i,j]
    for i in range(ntrain):
        for j in range(ndim):
            __training_points[ndim*i+j] = training_points[i,j]
    for i in range(ntrain):
        __weights[i] = weights[i]
    for i in range(nquery):
        __result[i] = 0

    cuda_evaluate(__query_points, __training_points, __weights, __metric, 
                  __kernel, bandwidth, __result, nquery, ntrain, ndim)

    result = numpy.zeros(nquery)
    for i in range(nquery):
        result[i] = __result[i]
    free(__query_points)
    free(__training_points)
    free(__weights)
    free(__result)
    return result*coeff/weights.sum()
