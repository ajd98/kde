/*
 * kernels.c
 *
 * Non-negative integrable functions for use in kernel density estimation.
 *
 * Includes the following kernels:
 *   - bump
 *   - cosine
 *   - epanechnikov
 *   - gaussian
 *   - logistic
 *   - quartic
 *   - tophat
 *   - triangle
 *   - tricubic
 *
 *
 * Written 18.03.04 by Alex DeGrave
 *
 */
#include <math.h>
#include <stdlib.h>
#include "kernels.h"

// Bump (smooth, compact support) kernel
// gaussian mapped to unit disk
//   must multiply by [exp(-1/2)*(K1(1/2) - K0(1/2))]**-1
//   where K1 and K0 are modified Bessel functions of the second kind, of
//   integer orders 1 and 0, respectively
double
bump(double x)
{
  if (fabs(x) < 1){
    return exp(1/(x*x-1));
  } else {
    return 0;
  }
}

// Cosine
//   must multiply by pi/4
double 
cosine_kernel(double x)
{
  if (fabs(x) < 1) {
    return cos(M_PI/2*x);
  } else {
    return 0;
  }
}

// Epanechnikov
//   must multiply by 3/4
double 
epanechnikov(double x)
{
  if (fabs(x) < 1) {
    return 1-x*x;
  } else {
    return 0;
  }
}

// Gaussian kernel
//   must multiply by 1/sqrt(2*pi)
double 
gaussian(double x)
{
  return exp(-x*x/2);
}

// Logistic
//   don't need to multiply by anything
double 
logistic(double x)
{
  return 1/(exp(-x)+2+exp(x));
}

// Quartic
//   must multiply by 15/16
double 
quartic(double x)
{
  if (fabs(x) < 1) {
    double y = 1-x*x;
    return y*y;
  } else {
    return 0;
  }
}

// Uniform kernel
//   must multiply by 0.5
double 
tophat(double x)
{
  if (fabs(x) < 1){
    return 1;
  } else{
    return 0;
  }
}

// Triangle kernel
//   don't need to multiply by anything
double
triangle(double x)
{
  double y = fabs(x);
  if (y < 1) {
    return 1-y;
  } else {
    return 0;
  }
}

// Tricubic kernel (compact, two derivatives)
//   must multiply by 70/81
double
tricube(double x)
{
  double y = fabs(x);
  if (y < 1) {
    double z = 1-y*y*y;
    return z*z*z;
  } else {
    return 0;
  }
}

