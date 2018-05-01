#ifndef DISTANCE_H
#define DISTANCE_H
extern __device__ double euclidean_distance(const double* u, const double* v, int n);
extern __device__ double euclidean_distance_ntorus(const double* u, const double* v, int n);
#endif // DISTANCE_H
