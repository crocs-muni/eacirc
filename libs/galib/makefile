# -*- Mode: makefile -*-
# Makefile for GAlib
# Copyright (c) 1994-1996 Massachusetts Institute of Technology
# Copyright (c) 1996-2005 Matthew Wall, all rights reserved
#
# If you need to customize the build of galib, you should first modify the
# variables in the makevars file.
GALIB_VERSION=2.4.7
GALIB_VER=247
TMPDIR=/var/tmp
RELDIR=$(TMPDIR)/galib$(GALIB_VER)
# -----------------------------------------------------------------------------

include makevars
include makefile.sources

INC_DIRS= -I..

.SUFFIXES: .cpp
.cpp.o:
	$(CXX) $(CXXFLAGS) $(INC_DIRS) -c $<

$(LIB): $(OBJS) 
	$(PRELINK)
	$(AR) $(LIB) $?
	$(RANLIB) $(LIB)
	echo "$(CXX) $(CXXFLAGS)" > BUILD
	@echo $(LIB) is now up-to-date
.C.a:;

clean:
	$(RM) $(LIB) BUILD
	$(RM) *.o 

install:: $(LIB)
	$(INSTALL) $(LIB) $(LIB_DEST_DIR)
	$(RM) $(HDR_DEST_DIR)/ga; $(MKDIR) $(HDR_DEST_DIR)/ga
	$(CP) $(HDRS) $(TMPL_SRCS) ../VERSION BUILD $(HDR_DEST_DIR)/ga

uninstall:
	$(RM) $(LIB_DEST_DIR)/$(LIB)
	$(RM) $(HDR_DEST_DIR)/ga

depend:
	$(MKDEPEND) $(INC_DIRS) $(SRCS)

# DO NOT DELETE THIS LINE -- make depend depends on it.
