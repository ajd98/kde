#ifndef DISTANCE_H
#define DISTANCE_H
__device__ double euclidean_distance(double* u, double* v, int n);
__device__ double euclidean_distance_ntorus(double* u, double* v, int n);
#endif // DISTANCE_H
