#*********************************************************************************
# EACirc project (makefile)
#
# This makefile is for UNIX platforms only (due to shell commands and paths format).
# Makefiles for GAlib and tinyXML can be used on other platforms, see details below.
#
# EACirc: unix environment required, compilation settings below
#         g++ version 4.7 or higher is needed
#         clang 3.4 or higher
#         compiler chosen according to system settings in vars CC and GCC
# GAlib: Compilation settings and platform dependent variables are in galib/makefile.
# tinyXML: Compilation settings and platform dependent variables are in tinyXML/makefile.
#
# DEBUG can be set to YES to include debugging info, or NO otherwise
DEBUG		= NO
# PROFILE can be set to YES to include profiling info, or NO otherwise
PROFILE		= NO
#*********************************************************************************

# complation settings
CXXFLAGS		= -std=c++11
DEBUG_FLAGS		= -g -DDEBUG
RELEASE_FLAGS	= -O3
PROFILE_FLAGS	= -p

# other global settings
INC_DIRS=-IEACirc -IEACirc/galib -IEACirc/tinyXML
INC_LIBS=-LEACirc/galib -LEACirc/tinyXML -lga -ltinyXML -lcrypto

# === EACirc Main ===
SOURCES=
HEADERS=
# libs and source (loaded from Qt project file)
include EACirc.pro
OBJECTS_MAIN_TEMP:=$(SOURCES:.cpp=.ocpp)
OBJECTS_MAIN:=$(OBJECTS_MAIN_TEMP:.c=.oc)

# rules and targets
ifeq (YES, ${DEBUG})
   FLAGS     += $(DEBUG_CXXFLAGS)
else
   FLAGS     += $(RELEASE_CXXFLAGS)
endif
ifeq (YES, $(PROFILE))
   FLAGS += $(PROFILE_CXXFLAGS)
endif

.PHONY: all
all: libs main

.PHONY: libs
libs: libga.a libtinyXML.a

libga.a:
	cd EACirc/galib && $(MAKE)
	@echo =====================================
	@echo === GAlib was successfully built. ===
	@echo =====================================

libtinyXML.a:
	cd EACirc/tinyXML && $(MAKE)
	@echo =======================================
	@echo === tinyXML was successfully built. ===
	@echo =======================================

%.ocpp: %.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS) $(INC_DIRS) -c -o "$@" "$<"

%.oc: %.c
	$(CC) $(FLAGS) $(INC_DIRS) -c -o "$@" "$<"

.PHONY: preBuild
preBuild:
	scripts/pre-build.sh

main: libs preBuild $(OBJECTS_MAIN)
	$(CXX) $(CXXFLAGS) $(FLAGS) -o EACirc/EACirc $(OBJECTS_MAIN) $(INC_DIRS) $(INC_LIBS)
	scripts/post-build.sh
	@echo ======================================
	@echo === EACirc was successfully built. ===
	@echo ======================================

clean: cleanmain

cleanall: cleanlibs cleanmain cleanresults

cleanresults:
	scripts/clean-results.sh
	@echo ==========================================
	@echo === Result files successfully cleaned. ===
	@echo ==========================================

cleanmain:
	rm -f $(OBJECTS_MAIN) EACirc/EACirc
	@echo =====================================================
	@echo === EACirc main build files successfully cleaned. ===
	@echo =====================================================

cleanlibs:
	cd EACirc/galib && $(MAKE) clean
	cd EACirc/tinyXML && $(MAKE) clean
	@echo ===============================================
	@echo === GAlib and tinyXML successfully cleaned. ===
	@echo ===============================================

test: all
	cd run && ./EACirc -log -test
	scripts/clean-results.sh
