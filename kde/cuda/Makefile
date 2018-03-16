# Makefile for cuda kde

all: libcukde.a

libcukde.a: 
	nvcc -lib --compiler-options '-fPIC' -arch=compute_30 -O3 -o $@ _evaluate.cu  -I../../include/

uninstall:
	rm *.a 

