#ifndef KERNELS_H
#define KERNELS_H
extern __device__ double bump(double x);
extern __device__ double cosine_kernel(double x);
extern __device__ double epanechnikov(double x);
extern __device__ double gaussian(double x);
extern __device__ double logistic(double x);
extern __device__ double quartic(double x);
extern __device__ double tophat(double x);
extern __device__ double triangle(double x);
extern __device__ double tricube(double x);
#endif // KERNELS_H
