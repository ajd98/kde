# cython: profile=True 
#
import numpy
cimport scipy.special.cython_special

from libc.math cimport pow

hyp1f2 = scipy.special.cython_special.hyp1f2
gamma = scipy.special.cython_special.gamma

cdef extern from "math.h":
    double exp(double x) nogil


cdef double S_n(int n, double r) nogil:
    '''
    The surface area of the sphere in n-dimensions
    '''
    return 2*pow(M_PI,(0.5*n))*pow(r,(n-1))/gamma(0.5*n)

cdef double _bump_1d_integrand(double r, int n) nogil:
    return exp(1/(pow(r,2)-1))*S_n(n,r)

def bump_coefficient(n):
    return scipy.integrate.quad(_bump_1d_integrand, 0, 1, args=(n,))

def cosine_coefficient(n):
    return 2*M_PI**(0.5*n)*hyp1f2(0.5*n,0.5,0.5*n+1, -1*M_PI**2/16)/gamma(0.5*n)/n

def epanechnikov_coefficient(n):
    return M_PI**(0.5*n)/gamma(0.5*n+2)

def gaussian_coefficient(n):
    return (2*numpy.pi)**(0.5*n)

cdef double _logistic_1d_integrand(double r, int n) nogil:
    return 1/(exp(-1*r)+2+exp(r))*S_n(n, r)

def logistic_coefficient(n):
    return scipy.integrate.quad(_logistic_1d_integrand)

def quartic_coefficient(n):
    return 16*M_PI**(0.5*n)/(n*(n*n+6*n+8)*gamma(0.5*n))

def tophat_coefficient(n):
    return 2*M_PI**(0.5*n)/(n*gamma(0.5*n))

def triangle_coefficient(n):
    return M_PI**(0.5*n)/((n+1)*gamma(0.5*n+1))

def tricube_coefficient(n):
    return 324*M_PI**(0.5*n)/(n*(n+3)*(n+6)*(n+9)*gamma(0.5*n))

