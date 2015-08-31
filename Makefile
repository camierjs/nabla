#######################
# MacOS||Linux switch #
#######################
ifeq ($(shell uname),Darwin)
CMAKE_ROOT_PATH=/opt/local/bin
COMPILER_ROOT_PATH=/opt/local/bin
COMPILER_POSTFIX=-mp-4.9
else
CMAKE_ROOT_PATH = /usr/bin
COMPILER_ROOT_PATH=/usr/bin
#COMPILER_ROOT_PATH=/usr/local/gcc/4.9.2/bin
COMPILER_POSTFIX=
endif

####################
# COMPILER OPTIONS #
# gcc -dM -E - < /dev/null
####################
C_FLAGS = -std=c99
MAKEFLAGS = --no-print-directory
export CC  = $(COMPILER_ROOT_PATH)/gcc$(COMPILER_POSTFIX)
export CXX = $(COMPILER_ROOT_PATH)/g++$(COMPILER_POSTFIX)
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
BUILD_MKDIR = mkdir -p $(BUILD_PATH) && sync && sync
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
	(cd $(BUILD_PATH)/tests && $(CTEST) --schedule-random -j $(NUMBR_PROCS))
test:tst
tst1:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j 1)
tstn:
	(cd $(BUILD_PATH)/tests && $(CTEST) -N)
tstg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS) -R gen)

# OKINA tests #
tstoua:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_upwindAP_run_1_std_seq)
tstou:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_upwind_run_1_std_seq)
tstol:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_1_sse_seq)

# LAMBDA tests #
tstl:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda)
tstll:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_lulesh_run_1_seq)
tstllo:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_lulesh_run_1_omp)
tstlau:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_upwindAP_run_1_seq)


# ARCANE tests #
tstau:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_upwind_gen_1)
tstagram:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_gram_gen_1)
tstagad:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_gad_run_1)
tstacomd:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_comd_run_1)
tstas:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_schrodinger_run_1)
tstas4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_schrodinger_run_4)
tstal1:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_lulesh_run_1)
tstag1:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_glace_run_1)
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

# CUDA tests #
tstcu:
#	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_gram_gen_1)
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
