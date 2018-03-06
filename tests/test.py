#!/usr/bin/env python
import sys
sys.path.append('../')
sys.path.append('../kde')
import kde.evaluate
import numpy
import scipy.integrate

query_points = numpy.linspace(-10,10,1001)[:,numpy.newaxis]
training_points = numpy.zeros(1)[numpy.newaxis,:]

def f_gaussian_1d(x):
    x = numpy.array((x,))[numpy.newaxis, :]
    return kde.evaluate.estimate_pdf_brute(x, training_points)[0]

def f_bump_1d(x):
    x = numpy.array((x,))[numpy.newaxis, :]
    return kde.evaluate.estimate_pdf_brute(x, training_points, kernel='bump')[0]

def f_cosine_1d(x):
    x = numpy.array((x,))[numpy.newaxis, :]
    return kde.evaluate.estimate_pdf_brute(x, training_points, kernel='cosine')[0]

def f_epanechnikov_1d(x):
    x = numpy.array((x,))[numpy.newaxis, :]
    return kde.evaluate.estimate_pdf_brute(x, training_points, kernel='epanechnikov')[0]

def f_logistic_1d(x):
    x = numpy.array((x,))[numpy.newaxis, :]
    return kde.evaluate.estimate_pdf_brute(x, training_points, kernel='logistic')[0]

def f_quartic_1d(x):
    x = numpy.array((x,))[numpy.newaxis, :]
    return kde.evaluate.estimate_pdf_brute(x, training_points, kernel='quartic')[0]

def f_tophat_1d(x):
    x = numpy.array((x,))[numpy.newaxis, :]
    return kde.evaluate.estimate_pdf_brute(x, training_points, kernel='tophat')[0]

def f_triangle_1d(x):
    x = numpy.array((x,))[numpy.newaxis, :]
    return kde.evaluate.estimate_pdf_brute(x, training_points, kernel='triangle')[0]

def f_tricube_1d(x):
    x = numpy.array((x,))[numpy.newaxis, :]
    return kde.evaluate.estimate_pdf_brute(x, training_points, kernel='tricube')[0]

print(scipy.integrate.quad(f_gaussian_1d, -10, 10)[0])
print(scipy.integrate.quad(f_bump_1d, -10, 10)[0])
print(scipy.integrate.quad(f_cosine_1d, -10, 10)[0])
print(scipy.integrate.quad(f_epanechnikov_1d, -10, 10)[0])
print(scipy.integrate.quad(f_logistic_1d, -10, 10)[0])
print(scipy.integrate.quad(f_quartic_1d, -10, 10)[0])
print(scipy.integrate.quad(f_tophat_1d, -10, 10)[0])
print(scipy.integrate.quad(f_triangle_1d, -10, 10)[0])
print(scipy.integrate.quad(f_tricube_1d, -10, 10)[0])

