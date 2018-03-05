# cython: profile=True 
#
import numpy
cimport numpy

cdef extern from "distance.h":
    double euclidean_distance(double* u, double* v, int size) nogil
    double euclidean_distance_ntorus(double* u, double* v, int size) nogil

cdef extern from "kernels.h":
    double bump(double x) nogil
    double cosine_func(double x) nogil
    double epanechnikov(double x) nogil
    double gaussian(double x) nogil
    double logistic(double x) nogil
    double quartic(double x) nogil
    double tophat(double x) nogil
    double triangle(double x) nogil
    double tricube(double x) nogil

cdef void _estimate_pdf_brute(double [:,:] query_points, 
                              double [:,:] training_points, 
                              double (*)(double *, double *, int) metric,
                              double (*)(double) kernel_func, 
                              double[:] result,
                              int nquery,
                              int ntrain):
    '''
    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(kernel_func(metric_func(x-xi)))
    '''
    with nogil:
        for i in range(nquerypts):
            for j in range(ntrainingpts):
                result[i] += kernel_func(metric(query_points[i], 
                                                training_points[j]))
    return result

def estimate_pdf_brute(query_points, training_points, metric='euclidean_distance', kernel='gaussian'):
    '''
    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(kernel_func(metric_func(x-xi)))
    '''
    # Parse keyword arguments
    if kernel == 'bump':
        kernel_func = bump
    elif kernel == 'cosine':
        kernel_func = cosine_func
    elif kernel == 'epanechnikov':
        kernel_func = epanechnikov
    elif kernel == 'gaussian':
        kernel_func = gaussian
    elif kernel == 'logistic':
        kernel_func = logistic
    elif kerenl == 'quartic':
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

    # Make memoryviews for use with nogil
    cdef double [:,:] _query_points = query_points
    cdef double [:,:] _training_points = training_points
    cdef double [:] _result = numpy.zeros(query_points.shape[0])
    cdef int nquery = query_points.shape[0]
    cdef int ntrain = training_points.shape[0]

    _estimate_pdf_brute(_query_points, _training_points, metric_func, 
                        kernel_func, _result, nquery, ntrain)

    return numpy.asarray(_result)
