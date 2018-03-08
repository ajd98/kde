# distance.pyx
from libc.math import fmod, sqrt

# Euclidean distance
cdef double euclidean_distance(double[:] v, double[:] u, int size) nogil:
    cdef double result = 0
    cdef double diff
    cdef int i
    for i in range(size): 
        diff = v[i] - u[i]
        result += diff*diff
    return sqrt(result)

# Euclidean distance in S1 x S1 x ... x S1 (an n-Torus)
# points are equivalent mod 360
cdef double euclidean_distance_ntorus(double* v, double* u, int size) nogil:
    cdef double result = 0
    cdef double diff
    cdef int i
    for i in range(size) {
        diff = v[i] - u[i]
        diff = fmod(diff, 360)
        diff -= 180
        result += diff*diff
    return sqrt(result)
