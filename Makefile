# Makefile for kde package
include Makefile.inc

SUBDIRS := kde/evaluate

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

uninstall: $(SUBDIRS)

.PHONY: $(SUBDIRS)
