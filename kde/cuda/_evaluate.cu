#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

// Types for device function pointers
typedef double (*METRICFUNC_t)(const double*, const double*, int)
typedef double (*KERNELFUNC_t)(double)

// Cuda kernel for evaluating kernel density estimate
__global__ void
_evaluateCu(const double* query_points, 
            const double* training_points, 
            METRICFUNC_t metric,
            KERNELFUNC_t kernel_func,
            double h, 
            double* result,
            int nquery, int ntrain, int ndim)
{
  // query_points and training_points are "2d" arrays indexed as
  //    "query_points[i,j]" = query_points[ndim*i+j]

  // Each thread operates on its own element of query_points
  int iquery = blockDim.x * blockIdx.x + threadIdx.x;
  int itrain;
  if (iquery<nquery) {
    for (itrain=0;itrain<ntrain;itrain++) {
      result[iquery] += kernel(distance(training_points[itrain], query_points[iquery])/h);
    }
    result[iquery] /= h;
  }
}

// Wrapper for cuda kernel
_evaluate(const double* query_points,
          const double* training_points, 
          METRICFUNC_t metric,
          KERNELFUNC_t kernel_func,
          double h, 
          double* result,
          int nquery, int ntrain, int ndim)
{
  // query_points and training_points are "2d" arrays indexed as
  //    "query_points[i,j]" = query_points[ndim*i+j]
  // Allocate device memory 

  double *d_query_points = NULL;
  err = cudaMalloc((void **)&d_query_points, ndim*nquery*sizeof(double));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device array ``query_points`` (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  double *d_training_points = NULL;
  err = cudaMalloc((void **)&d_training_points, ndim*ntrain*sizeof(double));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device array ``training_points`` (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  double *d_result = NULL;
  err = cudaMalloc((void **)&d_result, nquery*sizeof(double));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device array ``result`` (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy ``query_points`` and ``training_points`` into device memory
  err = cudaMemcpy(d_query_points, query_points, nquery*sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy array ``query_points`` from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_training_points, training_points, ntrain*sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy array ``training_points`` from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Initialize ``result`` and ``d_result`` to 0
  int i;
  for (i=0;i<nquery;i++){
    result
  }
}

