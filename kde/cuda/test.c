#include "_evaluate.cuh"
#include <stdlib.h>
#include <stdio.h>

int
main (void)
{
  int nquery = 5;
  int ntrain = 4;
  int ndim = 2;
  double h = 1;
  double* query_points = (double*)malloc(nquery*ndim*sizeof(double));
  double* training_points = (double*)malloc(ntrain*ndim*sizeof(double));
  double* weights = (double*)malloc(ntrain*sizeof(double));
  double* result = (double*)malloc(nquery*sizeof(double));
  char metric_s[]="euclidean_distance";
  char kernel_s[]="gaussian";

  int i;
  for(i=0;i<nquery;i++){
    query_points[ndim*i] = i;
    query_points[ndim*i+1] = i;
  }
  for(i=0;i<ntrain;i++){
    training_points[ndim*i] = i;
    training_points[ndim*i+1] = i;
  }
  for(i=0;i<ntrain;i++){
    weights[i] = 1./ntrain;
  }
  for(i=0;i<nquery;i++){
    result[i] = 0;
  }

  cuda_evaluate(query_points,
                training_points, 
                weights,
                metric_s,
                kernel_s,
                h, 
                result,
                nquery, ntrain, ndim);
  printf("success\n");
}
