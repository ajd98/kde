# Makefile for kde
include ../../Makefile.inc

OBJ = distance.o kernels.o

all: libkde.a

%.o: %.c 
	$(CC) -c -O3 -o $@ $< $(CFLAGS)

libkde.a: $(OBJ)
	ar cr libkde.a $(OBJ)

clean:
	-rm *.so *.a *.o

.PHONY: all clean
