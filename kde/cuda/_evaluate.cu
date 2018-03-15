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
void
_evaluate(const double* query_points,
          const double* training_points, 
          char* metric_s,
          char* kernel_func_s,
          double h, 
          double* result,
          int nquery, int ntrain, int ndim)
{
  // query_points and training_points are "2d" arrays indexed as
  //    "query_points[i,j]" = query_points[ndim*i+j]

  // Parse ``metric_s`` and ``kernel_func_s``
  METRICFUNC_t metric;
  if strcmp(metric_s, "euclidean_distance") {
    metric = euclidean_distance;
  } else if strcmp(metric_s, "eucldiean_distance_ntorus") {
    metric = euclidean_distance_ntorus;
  } else {
    printf("Metric ``%s`` not implemented.", metric_s);
    exit(EXIT_FAILURE);
  }
  
  KERNELFUNC_t kernel_func;
  if strcmp(kernel_func_s, "bump") {
    kernel_func = bump;
  } else if strcmp(kernel_func_s, "cosine") {
    kernel_func = cosine_func;
  } else if strcmp(kernel_func_s, "epanechnikov") {
    kernel_func = epanechnikov;
  } else if strcmp(kernel_func_s, "gaussian") {
    kernel_func = gaussian;
  } else if strcmp(kernel_func_s, "logistic") {
    kernel_func = logistic;
  } else if strcmp(kernel_func_s, "quartic") {
    kernel_func = quartic;
  } else if strcmp(kernel_func_s, "tophat") {
    kernel_func = tophat;
  } else if strcmp(kernel_func_s, "triangle") {
    kernel_func = triangle;
  } else if strcmp(kernel_func_s, "tricube") {
    kernel_func = tricube;
  } else {
    printf("Kernel ``%s`` not implemented.", kernel_func_s);
    exit(EXIT_FAILURE);
  }


  // Allocate device memory 
  double *d_query_points = NULL;
  err = cudaMalloc((void **)&d_query_points, ndim*nquery*sizeof(double));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device array ``query_points`` (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  double *d_training_points = NULL;
  err = cudaMalloc((void **)&d_training_points, ndim*ntrain*sizeof(double));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device array ``training_points`` (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  double *d_result = NULL;
  err = cudaMalloc((void **)&d_result, nquery*sizeof(double));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device array ``result`` (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy ``query_points`` and ``training_points`` into device memory
  err = cudaMemcpy(d_query_points, query_points, nquery*sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy array ``query_points`` from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_training_points, training_points, ntrain*sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy array ``training_points`` from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Initialize ``result`` and ``d_result`` to 0
  int i;
  for (i=0;i<nquery;i++){
    result[i] = 0;
  }
  err = cudaMemcpy(d_result, result, nquery*sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy array ``result`` from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Run the cuda kernel
  int threadsPerBlock = 256;
  int blocksPerGrid =(nquery + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  _evaluateCu<<<blocksPerGrid, threadsPerBlock>>>(d_query_points, d_training_points, metric, kernel_func, h, d_result, nquery, ntrain, ndim);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the result vector from device to host
  err = cudaMemcpy(result, d_result, nquery*sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_query_points);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device array ``d_query_points`` (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_training_points);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device array ``d_training_points`` (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_result);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device array ``d_result`` (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

