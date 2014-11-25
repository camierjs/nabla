#############
# COMPILERS #
#############
#export CC=/usr/bin/gcc
#export CXX=/usr/bin/g++

export CC  = /usr/local/gcc/4.9.2/bin/gcc
export CXX = /usr/local/gcc/4.9.2/bin/g++

export C_FLAGS = -std=c99

#################
# CMAKE OPTIONS #
#################
CMAKE_PATH = /usr
#/local/opendev1/gcc/cmake/3.0.0
CMAKE = $(CMAKE_PATH)/bin/cmake
CTEST = $(CMAKE_PATH)/bin/ctest

####################
# MAKEFILE OPTIONS #
####################
MAKEFLAGS = --no-print-directory

#########
# PATHS #
#########
NABLA_PATH = ~nabla/root
BUILD_PATH = /tmp/$(USER)/nabla

############
# COMMANDS #
############
BUILD_MKDIR = mkdir --parent $(BUILD_PATH) && sync && sync
BUILD_CMAKE = cd $(BUILD_PATH) && $(CMAKE) $(NABLA_PATH)
NUMBR_PROCS = $(shell getconf _NPROCESSORS_ONLN)

##################
# BUILD Commands #
##################
.PHONY: all
all:
	@[ ! -d $(BUILD_PATH) ] && ($(BUILD_MKDIR) && $(BUILD_CMAKE)) || exit 0
	@cd $(BUILD_PATH) && make -j $(NUMBR_PROCS)

#################
# TEST Commands #
#################
tst:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS))
tst1:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j 1) #V -I 45,45)
tstn:
	(cd $(BUILD_PATH)/tests && $(CTEST) -N)
tstg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS) -R gen)
tstga2:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_mic_gen_1)
tstra2:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_mic_run_1)
tstro:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R run_omp)
tstrc:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R run_cilk)
tstr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -R run)
tstrv:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R run)
tstv:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V)

############
# CLEANING #
############
cln:;(cd $(BUILD_PATH) && make clean)

############
# RSYNCing #
############
rs:
	rsync -v /tmp/camierjs/nabla/tests/lulesh_mic/lulesh_mic_16_omp_avx* ~/tmp/
	@echo "OMP_NUM_THREADS=1 ./lulesh_mic_16_omp_avx2|more"
