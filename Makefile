# Makefile for kde package
include Makefile.inc

SUBDIRS := kde/evaluate
CU_SUBDIRS := kde/cuda

all: $(SUBDIRS) _evaluate.so

_evaluate.so: 
	python setup.py build_ext --inplace

cuda: all $(CU_SUBDIRS)
	python setup_cu.py build_ext --inplace

$(SUBDIRS):
	$(MAKE) -C $@

$(CU_SUBDIRS):
	$(MAKE) -C $@

clean: 
	git clean -fd

.PHONY: $(SUBDIRS) $(CU_SUBDIRS) cuda all clean
