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
    double tricubic(double x) nogil

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

def estimate_pdf_brute(query_points, training_points, metric, kernel_func='gaussian'):
    '''
    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(kernel_func(metric_func(x-xi)))
    '''
    if kernel_func == 'gaussian':
        result = _estimate_pdf_brute(query_points, training_points, metric, gaussian):
    if kernel_func == 'bump':
        result = _estimate_pdf_brute(query_points, training_points, metric, bump):
    if kernel_func == 'tricubic':
        result = _estimate_pdf_brute(query_points, training_points, metric, tricubic):
    if kernel_func == 'tophat':
        result = _estimate_pdf_brute(query_points, training_points, metric, tophat)
    return result

                
def estimate_pdf_gaussian(query_points, training_points, metric):
    '''
    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(g(metric_func(x-xi)))

    where
          g(y) = 1/(2*pi*h)*exp(-(y)**2/(2*h))
    '''
    return estimate_pdf_brute(query_points, training_points, metric, gaussian)

def estimate_pdf_smooth_compact(query_points, training_points, metric):
    '''
    Evaluate the kernel density estimate at ``query_points`` as:

          f_hat(x) = 1/n sum(g(metric_func(x-xi)))

    where
          g(y) = cos(
    '''
