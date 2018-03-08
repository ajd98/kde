# cython: boundscheck=False
#
import numpy
cimport numpy
cimport kde.evaluate.kernel_coefficients as kernel_coefficients
from libc.stdlib cimport malloc, free

COMPACT_KERNELS=['bump', 'cosine', 'epanechnikov', 'quartic', 'tophat', 
                 'triangle', 'tricube']

cdef extern from "distance.h":
    double euclidean_distance(double* u, double* v, int size) nogil
    double euclidean_distance_ntorus(double* u, double* v, int size) nogil

cdef extern from "kernels.h":
    double bump(double x) nogil
    double cosine_kernel(double x) nogil
    double epanechnikov(double x) nogil
    double gaussian(double x) nogil
    double logistic(double x) nogil
    double quartic(double x) nogil
    double tophat(double x) nogil
    double triangle(double x) nogil
    double tricube(double x) nogil

ctypedef double (*METRICFUNC_t)(double *, double*, int) nogil
ctypedef double (*KERNELFUNC_t)(double) nogil

cdef void _estimate_pdf_brute(double [:,:] query_points, 
                              double [:,:] training_points, 
                              METRICFUNC_t metric,
                              KERNELFUNC_t kernel_func, 
                              double h,
                              double[:] result,
                              int nquery,
                              int ntrain,
                              int ndim):
    '''
    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(kernel_func(metric_func(x-xi)))
    '''
    cdef double* x  = <double *>malloc(ndim*sizeof(double))
    cdef double* xi = <double *>malloc(ndim*sizeof(double))
    with nogil:
        for i in range(nquery):
            for j in range(ntrain):
                for k in range(ndim):
                    x[k] = query_points[i,k]
                    xi[k] = training_points[j,k]
                result[i] += kernel_func(metric(x, xi, ndim)/h)
    free(x)
    free(xi)
    return

cdef void _estimate_pdf_brute_weighted(double [:,:] query_points, 
                                       double [:,:] training_points, 
                                       double [:] weights, 
                                       METRICFUNC_t metric,
                                       KERNELFUNC_t kernel_func, 
                                       double h,
                                       double[:] result,
                                       int nquery,
                                       int ntrain,
                                       int ndim):
    '''
    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = sum(w*kernel_func(metric_func(x-xi)))
    '''
    cdef double* x  = <double *>malloc(ndim*sizeof(double))
    cdef double* xi = <double *>malloc(ndim*sizeof(double))
    with nogil:
        for i in range(nquery):
            for j in range(ntrain):
                for k in range(ndim):
                    x[k] = query_points[i,k]
                    xi[k] = training_points[j,k]
                result[i] += weights[j]*kernel_func(metric(x, xi, ndim)/h)
    free(x)
    free(xi)
    return

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

    # Parse keyword arguments
    if kernel == 'bump':
        kernel_func = bump
        coeff = kernel_coefficients.bump_coefficient(ndim)
    elif kernel == 'cosine':
        kernel_func = cosine_kernel
        coeff = kernel_coefficients.cosine_coefficient(ndim)
    elif kernel == 'epanechnikov':
        kernel_func = epanechnikov
        coeff = kernel_coefficients.epanechnikov_coefficient(ndim)
    elif kernel == 'gaussian':
        kernel_func = gaussian
        coeff = kernel_coefficients.gaussian_coefficient(ndim)
    elif kernel == 'logistic':
        kernel_func = logistic
        coeff = kernel_coefficients.logistic_coefficient(ndim)
    elif kernel == 'quartic':
        kernel_func = quartic
        coeff = kernel_coefficients.quartic_coefficient(ndim)
    elif kernel == 'tophat':
        kernel_func = tophat
        coeff = kernel_coefficients.tophat_coefficient(ndim)
    elif kernel == 'triangle':
        kernel_func = triangle
        coeff = kernel_coefficients.triangle_coefficient(ndim)
    elif kernel == 'tricube':
        kernel_func = tricube
        coeff = kernel_coefficients.tricube_coefficient(ndim)
    else:
        raise ValueError("Kernel {:s} not found.".format(kernel))
    if metric == 'euclidean_distance':
       metric_func = euclidean_distance
    if metric == 'euclidean_distance_ntorus':
       metric_func = euclidean_distance_ntorus
       if not kernel in COMPACT_KERNELS:
           raise ValueError("Kernel {:s} is not compact, and therefore invalid "
                            "for use with n-torus space.".format(kernel))
       elif bandwidth > 180:
           raise ValueError("Bandwidth {:f} is too large for use with n-torus "
                            "space. Bandwidth must be less than or equal to 180."\
                            .format(bandwidth))

    # Make memoryviews for use with nogil
    cdef double [:,:] _query_points = query_points
    cdef double [:,:] _training_points = training_points
    cdef double [:] _result = numpy.zeros(query_points.shape[0])

    if weights is None:
        _estimate_pdf_brute(_query_points, _training_points, metric_func, 
                            kernel_func, bandwidth, _result, nquery, ntrain, ndim)
        result = numpy.asarray(_result)*coeff/ntrain/bandwidth
        return result

    # Else
    cdef double[:] _weights = weights
    _estimate_pdf_brute_weighted(_query_points, _training_points, _weights,
                                 metric_func, kernel_func, bandwidth, 
                                 _result, nquery, ntrain, ndim)
    result = numpy.asarray(_result)*coeff/weights.sum()/bandwidth

    return result
