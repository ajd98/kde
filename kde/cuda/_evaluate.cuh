#ifndef EVALUATE_CU_H
#define EVALUATE_CU_H
void
_evaluate_cu(const double* query_points, 
             const double* training_points, 
             const double* weights,
             enum metricopt metric_idx,
             enum kernelopt kernel_idx,
             double h, 
             double* result,
             int nquery, int ntrain, int ndim);
#endif // EVALUATE_CU_H
