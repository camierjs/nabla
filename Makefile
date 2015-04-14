##############
# ROOT_PATHS #
##############
CMAKE_ROOT_PATH = /usr/bin
COMPILER_ROOT_PATH=/usr/bin
COMPILER_ROOT_PATH=/usr/local/gcc/4.9.2/bin

####################
# COMPILER OPTIONS #
####################
C_FLAGS = -std=c99
MAKEFLAGS = --no-print-directory
export CC  = $(COMPILER_ROOT_PATH)/gcc
export CXX = $(COMPILER_ROOT_PATH)/g++
export LD_LIBRARY_PATH=$(COMPILER_ROOT_PATH)/../lib64

#################
# CMAKE OPTIONS #
#################
CMAKE = $(CMAKE_ROOT_PATH)/cmake
CTEST = $(CMAKE_ROOT_PATH)/ctest

#########
# PATHS #
#########
NABLA_PATH = $(shell pwd)
BUILD_PATH = /tmp/nabla

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
all:
	@[ ! -d $(BUILD_PATH) ] && ($(BUILD_MKDIR) && $(BUILD_CMAKE)) || exit 0
	@cd $(BUILD_PATH) && make -j $(NUMBR_PROCS)

##################
# CONFIG Command #
##################
cfg:
	$(BUILD_CMAKE)
config:cfg

bin:
	@cd $(BUILD_PATH) && make install
install:bin

##################
# TESTs Commands #
##################
tst:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS))
test:tst
tst1:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j 1)
tstn:
	(cd $(BUILD_PATH)/tests && $(CTEST) -N)
tstg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS) -R gen)
tsta:
	(cd $(BUILD_PATH)/tests && $(CTEST) -R arcane)

tstas:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_schrodinger_run_1)
tstas4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_schrodinger_run_4)
tsthas:
	(cd $(BUILD_PATH)/tests && /usr/local/arcane/testing/bin/hyoda $(CTEST) -V -R nabla_arcane_schrodinger_run_1)
tsthas4:
	(cd $(BUILD_PATH)/tests && /usr/local/arcane/testing/bin/hyoda $(CTEST) -V -R nabla_arcane_schrodinger_run_4)

tsts:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_sethi_run_1)
tsths:
	(cd $(BUILD_PATH)/tests && /usr/local/arcane/testing/bin/hyoda $(CTEST) -V -R nabla_arcane_sethi_run_1)
tsts4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_sethi_run_4)

tstm:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_pDDFV_run_1)
tstmh:
	(cd $(BUILD_PATH)/tests && /usr/local/arcane/testing/bin/hyoda $(CTEST) -V -R nabla_arcane_pDDFV_run_1)
tstm4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_pDDFV_run_4)

tstr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -R run)
tstro:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R run_omp)
tstrc:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R run_cilk)
tstl:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_lulesharcane_run_1)
tstl4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_lulesharcane_run_4)
tstl8:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_lulesharcane_run_8)
tstseq:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_1_std_seq)
tstmh1:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_mhydro_run_1)
tststd:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_4_std_omp)
tststd4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_4_std_omp)
tstsse:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_4_sse_omp)
tstavx:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_4_avx_omp)
tstavx2:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_4_avx2_omp)
tstv:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j 1 -V)
tstcu:
#	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_mhydro_gen_1)
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_lulesh_run_1)

############
# CLEANING #
############
cln:;(cd $(BUILD_PATH) && make clean)
clean:cln

###########
# PHONIES #
###########
.PHONY: all cfg config bin tst cln clean
