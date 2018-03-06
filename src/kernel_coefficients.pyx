# cython: profile=True 
#
import numpy
import scipy.integrate
from scipy.special import hyp1f2, gamma

cimport numpy
from scipy.special.cython_special cimport gamma as _gamma
from libc.math cimport pow, M_PI


cdef extern from "math.h":
    double exp(double x) nogil


cdef double S_n(INT32_t n, double r) nogil:
    '''
    The surface area of the sphere in n-dimensions
    '''
    return 2*pow(M_PI,0.5*n)*pow(r,n-1.)/_gamma(0.5*n)

cdef double _bump_1d_integrand(double r, INT32_t n) nogil:
    return exp(1/(r*r-1))*S_n(n,r)

cpdef bump_coefficient(n):
    integral = scipy.integrate.quad(_bump_1d_integrand, 0, 1, args=(n,))[0]
    return 1/integral

cpdef cosine_coefficient(n):
    integral = 2*pow(M_PI,0.5*n)*hyp1f2(0.5*n,0.5,0.5*n+1, -1*M_PI**2/16)[0]/gamma(0.5*n)/n
    return 1/integral

cpdef epanechnikov_coefficient(n):
    integral = M_PI**(0.5*n)/gamma(0.5*n+2)
    return 1/integral

cpdef gaussian_coefficient(n):
    integral = (2*numpy.pi)**(0.5*n)
    return 1/integral

cdef double _logistic_1d_integrand(double r, INT32_t n) nogil:
    return 1/(exp(-1*r)+2+exp(r))*S_n(n, r)

cpdef logistic_coefficient(n):
    integral = scipy.integrate.quad(_logistic_1d_integrand, 0, 1000, args=(n,))[0]
    return 1/integral

cpdef quartic_coefficient(n):
    integral = 16*M_PI**(0.5*n)/(n*(n*n+6*n+8)*gamma(0.5*n))
    return 1/integral

cpdef tophat_coefficient(n):
    integral = 2*M_PI**(0.5*n)/(n*gamma(0.5*n))
    return 1/integral

cpdef triangle_coefficient(n):
    integral = 2*M_PI**(0.5*n)/((n+1)*n*gamma(0.5*n))
    return 1/integral

cpdef tricube_coefficient(n):
    integral = 324*M_PI**(0.5*n)/(n*(n+3)*(n+6)*(n+9)*gamma(0.5*n))
    return 1/integral
