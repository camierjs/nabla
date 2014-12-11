##############
# ROOT_PATHS #
##############
#COMPILER_ROOT_PATH=/usr
COMPILER_ROOT_PATH=/usr/local/gcc/4.9.2
CMAKE_ROOT_PATH = /usr

####################
# COMPILER OPTIONS #
####################
C_FLAGS = -std=c99
MAKEFLAGS = --no-print-directory
export CC  = $(COMPILER_ROOT_PATH)/bin/gcc
export CXX = $(COMPILER_ROOT_PATH)/bin/g++
export LD_LIBRARY_PATH=$(COMPILER_ROOT_PATH)/lib64

#################
# CMAKE OPTIONS #
#################
CMAKE = $(CMAKE_ROOT_PATH)/bin/cmake
CTEST = $(CMAKE_ROOT_PATH)/bin/ctest

#########
# PATHS #
#########
NABLA_PATH = $(shell pwd)
BUILD_PATH = /tmp/$(USER)/nabla

############
# COMMANDS #
############
BUILD_MKDIR = mkdir --parent $(BUILD_PATH) && sync && sync
CMAKE_FLAGS = --warn-uninitialized
BUILD_CMAKE = cd $(BUILD_PATH) && $(CMAKE) $(CMAKE_FLAGS) $(NABLA_PATH)
NUMBR_PROCS = $(shell getconf _NPROCESSORS_ONLN)

##################
# BUILD Commands #
##################
.PHONY: all
all:
	@[ ! -d $(BUILD_PATH) ] && ($(BUILD_MKDIR) && $(BUILD_CMAKE)) || exit 0
	@cd $(BUILD_PATH) && make -j $(NUMBR_PROCS)

##################
# CONFIG Command #
##################
cfg:
	$(BUILD_CMAKE)

bin:
	@cd $(BUILD_PATH) && make install

#################
# TEST Commands #
#################
tst:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS))
tst1:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j 1)
tstn:
	(cd $(BUILD_PATH)/tests && $(CTEST) -N)
tstg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS) -R gen)
tstr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -R run)
tstro:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R run_omp)
tstrc:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R run_cilk)
tstrv:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_1_std_omp)
tstv:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j 1 -V)

############
# CLEANING #
############
cln:;(cd $(BUILD_PATH) && make clean)
