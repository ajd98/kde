# kernel_coefficients.pxd
#
# Written 18.03.05 by Alex DeGrave

cimport numpy

ctypedef numpy.int32_t INT32_t
cdef double S_n(INT32_t n, double r) nogil
cdef double _bump_1d_integrand(double r, INT32_t n) nogil
cpdef bump_coefficient(n)
cpdef cosine_coefficient(n)
cpdef epanechnikov_coefficient(n)
cpdef gaussian_coefficient(n)
cdef double _logistic_1d_integrand(double r, INT32_t n) nogil
cpdef logistic_coefficient(n)
cpdef quartic_coefficient(n)
cpdef tophat_coefficient(n)
cpdef triangle_coefficient(n)
cpdef tricube_coefficient(n)
