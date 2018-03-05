# cython: profile=True 
#
import numpy
cimport numpy

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
                              double[:] result,
                              int nquery,
                              int ntrain,
                              int ndim):
    '''
    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(kernel_func(metric_func(x-xi)))
    '''
    with nogil:
        for i in range(nquery):
            for j in range(ntrain):
                result[i] += kernel_func(metric(&query_points[i][::1][0], 
                                                &training_points[j][::1][0],
                                                ndim))
    return

def estimate_pdf_brute(query_points, training_points, 
                       metric='euclidean_distance', kernel='gaussian'):
    '''
    estimate_pdf_brute(query_points, training_points, 
                       metric='euclidean_distance', kernel='gaussian')


    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(kernel_func(metric_func(x-xi)))

    ----------
    Parameters
    ----------
    query_points: (numpy.ndarray)
    training_points: (numpy.ndarray)
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
    # Parse keyword arguments
    if kernel == 'bump':
        kernel_func = bump
    elif kernel == 'cosine':
        kernel_func = cosine_kernel
    elif kernel == 'epanechnikov':
        kernel_func = epanechnikov
    elif kernel == 'gaussian':
        kernel_func = gaussian
    elif kernel == 'logistic':
        kernel_func = logistic
    elif kernel == 'quartic':
        kernel_func = quartic
    elif kernel == 'tophat':
        kernel_func = tophat
    elif kernel == 'triangle':
        kernel_func = triangle
    elif kernel == 'tricube':
        kernel_func = tricube
    else:
        raise ValueError("Kernel {:s} not found.".format(kernel))
    if metric == 'euclidean_distance':
       metric_func = euclidean_distance
    if metric == 'euclidean_distance_ntorus':
       metric_func = euclidean_distance_ntorus

    if not training_points.shape[1] != query_points.shape[1]:
        raise TypeError("Training points and query points must have same "
                        "dimension but have dimension {:d} and {:d}"\
                        .format(training_points.shape[1], query_points.shape[1]))

    # Make memoryviews for use with nogil
    cdef double [:,:] _query_points = query_points
    cdef double [:,:] _training_points = training_points
    cdef double [:] _result = numpy.zeros(query_points.shape[0])
    cdef int nquery = query_points.shape[0]
    cdef int ntrain = training_points.shape[0]
    cdef int ndim = training_points.shape[1]

    _estimate_pdf_brute(_query_points, _training_points, metric_func, 
                        kernel_func, _result, nquery, ntrain, ndim)

    return numpy.asarray(_result)
