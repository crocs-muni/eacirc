#*********************************************************************************
# EACirc project (makefile)
#
# This makefile is for UNIX platforms only (due to shell commands and paths format).
# Makefiles for GAlib and tinyXML can be used on other platforms, see details below.
#
# EACirc: unix environment required, compilation settings below.
#         g++ version 4.7 or higher is needed
# GAlib: Compilation settings and platform dependent variables are in galib/makefile.
# tinyXML: Compilation settings and platform dependent variables are in tinyXML/makefile.
#
# DEBUG can be set to YES to include debugging info, or NO otherwise
DEBUG		= NO
# PROFILE can be set to YES to include profiling info, or NO otherwise
PROFILE		= NO
# output name for the compiled and linked application
OUTNAME_MAIN	= eacirc
OUTNAME_CHECKER	= checker
# folder to put linked application and config file into
RUN_DIR		= run
#*********************************************************************************

# complation settings
#CXX			= g++
CXX			= g++-4.7
CC			= gcc-4.7
CXXFLAGS		= -std=c++11 # -Wall
DEBUG_FLAGS		= -g -DDEBUG
RELEASE_FLAGS	= -O3
PROFILE_FLAGS	= -p

# other global settings
INC_DIRS= -IEACirc -IEACirc/galib -IEACirc/tinyXML
INC_LIBS= -LEACirc/galib -LEACirc/tinyXML -lga -ltinyXML

# === EACirc Main ===
SOURCES=
HEADERS=
# libs and source (loaded from Qt project file)
include EACirc.pro
OBJECTS_MAIN_TEMP:=$(SOURCES:.cpp=.ocpp)
OBJECTS_MAIN:=$(OBJECTS_MAIN_TEMP:.c=.oc)

# === EACirc Checker ===
SOURCES=
HEADERS=
# libs and source (loaded from Qt project file)
include Checker.pro
OBJECTS_CHECKER:=$(SOURCES:.cpp=.ocpp)

# rules and targets
ifeq (YES, ${DEBUG})
   FLAGS     += $(DEBUG_CXXFLAGS)
else
   FLAGS     += $(RELEASE_CXXFLAGS)
endif
ifeq (YES, $(PROFILE))
   FLAGS += $(PROFILE_CXXFLAGS)
endif

all: libs main checker

libs:
	cd EACirc/galib && $(MAKE)
	@echo === GAlib was successfully built. ===
	cd EACirc/tinyXML && $(MAKE)
	@echo === tinyXML was successfully built. ===

%.ocpp: %.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) $(INC_DIRS) -c -o "$@" "$<"

%.oc: %.c
	$(CC) $(FLAGS) $(INC_DIRS) -c -o "$@" "$<"

main: libs $(OBJECTS_MAIN)
	mkdir -p $(RUN_DIR)
	$(CXX) $(CXXFLAGS) -o $(RUN_DIR)/$(OUTNAME_MAIN) $(OBJECTS_MAIN) $(INC_DIRS) $(INC_LIBS)
	if [ ! -f $(RUN_DIR)/config.xml ]; then cp EACirc/config.xml $(RUN_DIR)/; fi
	@echo === $(OUTNAME_MAIN) was successfully built. ===

checker: $(OBJECTS_CHECKER)
	mkdir -p $(RUN_DIR)
	$(CXX) $(CXXFLAGS) -o $(RUN_DIR)/$(OUTNAME_CHECKER) $(OBJECTS_CHECKER) $(INC_DIRS) $(INC_LIBS)
	@echo === $(OUTNAME_CHECKER) was successfully built. ===

cleanall: cleanresults cleanlibs cleanmain cleanchecker

cleanresults:
	cd $(RUN_DIR) && rm -f *.log *.txt *.bin *.c *.dot *.xml *.2
	@echo === Result files successfully cleaned. ===

cleanmain:
	rm -f $(OBJECTS_MAIN) $(RUN_DIR)/$(OUTNAME_MAIN)
	@echo === $(OUTNAME_MAIN) successfully cleaned. ===

cleanchecker:
	rm -f $(OBJECTS_CHECKER) $(RUN_DIR)/$(OUTNAME_CHECKER)
	@echo === $(OUTNAME_CHECKER) successfully cleaned. ===

cleanlibs:
	cd EACirc/galib && $(MAKE) clean
	cd EACirc/tinyXML && $(MAKE) clean
