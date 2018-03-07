# Makefile for kde
include ../../Makefile.inc

OBJ = distance.o kernels.o

all: _evaluate.so

%.o: %.c 
	$(CC) -c -O3 -o $@ $< $(CFLAGS)

libkde.a: $(OBJ)
	ar cr libkde.a $(OBJ)

_evaluate.so: libkde.a 
	python setup.py build_ext --inplace

uninstall:
	rm *.so *.a *.o

