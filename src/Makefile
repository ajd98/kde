# Makefile for volume calculation tools

CC=gcc
IDIR=../include
CFLAGS=-I$(IDIR) -Wall -L. -g -std=gnu99 -fPIC

_DEPS=queue.h 
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))

OBJ = distance.o kernels.o

%.o: %.c $(DEPS)
	$(CC) -c -O3 -o $@ $< $(CFLAGS)

libkde.a: $(OBJ)
	ar cr libkde.a $(OBJ)

kde: libkde.a 
	python setup.py build_ext --inplace
	mv *.so ../kde

install: kde

uninstall:
	if [ -a test ]; then\
      rm test ;\
	fi;
	rm *.so *.a *.o
