# Makefile for kde package
include Makefile.inc

SUBDIRS := kde/evaluate

all: $(SUBDIRS) _evaluate.so

_evaluate.so: 
	python setup.py build_ext --inplace

	

$(SUBDIRS):
	$(MAKE) -C $@

uninstall: $(SUBDIRS)

.PHONY: $(SUBDIRS)
