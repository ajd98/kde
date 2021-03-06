/*
 * dist.c
 */
#include <math.h>
#include <stdlib.h>

// arithmetic modulus
double 
arithmeticfmod(double x, double d)
{
  double angle = fmod(x, d) ;
  if (angle < 0) {
    angle += d;
  }
  return angle;
}

// Euclidean distance
double 
euclidean_distance(double* v, double* u, int size) 
{
  double result = 0;
  double diff;
  int i;
  for (i=0;i<size;i++) {
    diff = v[i] - u[i];
    result += diff*diff;
  }
  return sqrt(result);
}

// Euclidean distance in S1 x S1 x ... x S1 (an n-Torus)
// points are equivalent mod 360
double 
euclidean_distance_ntorus(double* v, double* u, int size) 
{
  double result = 0;
  double diff;
  int i;
  for (i=0;i<size;i++) {
    diff = v[i] - u[i];
    diff = arithmeticfmod(diff, 360);
    diff = fmin(fabs(diff), fabs(diff-360));
    result += diff*diff;
  }
  return sqrt(result);
}

