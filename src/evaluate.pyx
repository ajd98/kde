# cython: profile=True 
#
import numpy

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

def _estimate_pdf_brute(query_points, training_points, metric, kernel_func):
    '''
    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(kernel_func(metric_func(x-xi)))
    '''
    int nquerypts = query_points.shape[0]
    int ntrainingpts = training_points.shape[0]
    result = numpy.zeros(nquerypts)
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
    if kernel == 'bump':
        kernel_func = bump
    elif kernel == 'cosine':
        kernel_func == cosine_func
    elif kernel == 'epanechnikov':
        kernel_func == epanechnikov
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
    if metric == 'euclidean_distance'
       metric_func = euclidean_distance
    if metric == 'euclidean_distance_ntorus'
       metric_func = euclidean_distance_ntorus
    result = _estimate_pdf_brute(query_points, training_points, metric_func, kernel_func)
    return result
