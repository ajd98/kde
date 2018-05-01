# Makefile for cuda kde
NVCCFLAGS = -arch=compute_20 -shared --compiler-options '-fPIC'

all: libcukde.a

OBJ = distance.o kernels.o _evaluate.o

devicelib.o: $(OBJ)
	nvcc $(NVCCFLAGS) -lcuda -lcudart -dlink -o devicelib.o $(OBJ)

%.o: %.cu
	nvcc $(NVCCFLAGS) -O3 -I../../include/ -dc $@ $<

libcukde.a: devicelib.o
	ar cr libcukde.a $(OBJ) devicelib.o
	#gcc $(OBJ) devicelib.o -L /opt/cuda/lib64 -lcudart  -shared -o libcukde.so

clean:
	rm *.a *.o

.PHONY: all clean
