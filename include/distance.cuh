#ifndef DISTANCE_H
#define DISTANCE_H
__device__ double euclidean_distance(const double* u, const double* v, int n);
__device__ double euclidean_distance_ntorus(const double* u, const double* v, int n);
#endif // DISTANCE_H
