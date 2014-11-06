#############
# COMPILERS #
#############
export CC=/usr/local/opendev1/gcc/gcc/4.9.0/bin/gcc
export CXX=/usr/local/opendev1/gcc/gcc/4.9.0/bin/g++
export C_FLAGS = -std=c99

#################
# CMAKE OPTIONS #
#################
CMAKE_PATH = /usr/local/opendev1/gcc/cmake/3.0.0
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
	@[ ! -d $(BUILD_PATH) ] && ($(BUILD_MKDIR)&&$(BUILD_CMAKE)) || exit 0
	@cd $(BUILD_PATH) && make -j $(NUMBR_PROCS)

#################
# TEST Commands #
#################
tst:
	(cd $(BUILD_PATH)/tests && ctest)
tstn:
	(cd $(BUILD_PATH)/tests && ctest -N)
tstg:
	(cd $(BUILD_PATH)/tests && ctest -R gen)
tstga2:
	(cd $(BUILD_PATH)/tests && ctest -V -R nabla_okina_lulesh_mic_gen_1)
tstra2:
	(cd $(BUILD_PATH)/tests && ctest -V -R nabla_okina_lulesh_mic_run_1)
tstro:
	(cd $(BUILD_PATH)/tests && ctest -V -R run_omp)
tstrc:
	(cd $(BUILD_PATH)/tests && ctest -V -R run_cilk)
tstr:
	(cd $(BUILD_PATH)/tests && ctest -R run)
tst4:
	(cd $(BUILD_PATH)/tests && ctest -j 4)
tstv:
	(cd $(BUILD_PATH)/tests && ctest -V)

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
