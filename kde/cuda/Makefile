# Makefile for cuda kde
NVCCFLAGS = -arch=compute_30 -shared --compiler-options '-fPIC'

all: libcukde.a

OBJ = distance.o kernels.o _evaluate.o

devicelib.o: $(OBJ)
	nvcc $(NVCCFLAGS) -dlink -o devicelib.o $(OBJ)

%.o: %.cu
	nvcc $(NVCCFLAGS) -O3 -I../../include/ -dc $@ $<

libcukde.a: devicelib.o
	ar cr libcukde.a $(OBJ) devicelib.o

uninstall:
	rm *.a *.o

