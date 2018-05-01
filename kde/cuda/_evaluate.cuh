#ifndef EVALUATE_CU_H
#define EVALUATE_CU_H
extern "C" void
cuda_evaluate(const double* query_points,
              const double* training_points, 
              const double* weights,
              char* metric_s,
              char* kernel_s,
              double h, 
              double* result,
              int nquery, int ntrain, int ndim);
#endif // EVALUATE_CU_H
