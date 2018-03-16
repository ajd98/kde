# Makefile for cuda kde
all: libcukde.a

libcukde.a: 
	nvcc -lib --compiler-options '-fPIC' -arch=sm_20 -O3 -o $@ _evaluate.cu 

uninstall:
	rm *.a 

