#ifndef KERNELS_H
#define KERNELS_H
__device__ double bump(double x);
__device__ double cosine_kernel(double x);
__device__ double epanechnikov(double x);
__device__ double gaussian(double x);
__device__ double logistic(double x);
__device__ double quartic(double x);
__device__ double tophat(double x);
__device__ double triangle(double x);
__device__ double tricube(double x);
#endif // KERNELS_H
