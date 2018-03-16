# Makefile for cuda kde

all: libcukde.a

OBJ = distance.o kernels.o _evaluate.o

%.o: %.cu
	nvcc -c -O3 --compiler-options '-fPIC' -arch=compute_30 -I../../include/ -o $@ $< 

libcukde.a: $(OBJ)
	ar cr libcukde.a $(OBJ)

uninstall:
	rm *.a *.o

