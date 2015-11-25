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
#export LD_LIBRARY_PATH=$(COMPILER_ROOT_PATH)/../lib64:${LD_LIBRARY_PATH}

#################
# CMAKE OPTIONS #
#################
CMAKE = $(CMAKE_ROOT_PATH)/cmake
CTEST = $(CMAKE_ROOT_PATH)/ctest
HYODA = /usr/local/arcane/testing/bin/hyoda

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
	(cd $(BUILD_PATH)/tests && $(CTEST) --schedule-random -j 4) # --schedule-random -j $(NUMBR_PROCS))
test:tst
tst1:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j 1)
tstn:
	(cd $(BUILD_PATH)/tests && $(CTEST) -N)
tstg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS) -R gen)

###############
# OKINA tests #
###############
tstoua:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_upwindAP_run_1_std_seq)
tstou:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_upwind_run_1_std_seq)
tstolr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_1_sse_seq)
tstoa:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_aecjs_gen_1)

################
# LAMBDA tests #
################
tstl:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda)
tstllr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_lulesh_run_1_seq)
tstllo:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_lulesh_run_1_omp)
tstlau:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_upwindAP_run_1_seq)
tstlhg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_heat_gen_1_seq)
tstlhr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_heat_run_1_seq)
tstlar:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_aleph1D_run_1_seq)
tstlbr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_amber_run_1_seq)
tstlxr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_deflex_run_1_seq)
tstldr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_darcy_run_1_seq)
tstlgg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_gram_gen_1)

################
# ARCANE tests #
################
tstamr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_mhydro_run_1)
tstaeg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_aecjs_gen_1)
tstadg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_darcy_gen_1)
tstadr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_darcy_run_1)
tstadr4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_darcy_run_4)
tstadrh:
	(cd $(BUILD_PATH)/tests && $(HYODA) $(CTEST) -V -R nabla_arcane_darcy_run_1)
tstasw:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_swirl_gen_1)
tstabg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_amber_gen_1)
tstaag:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_aleph1D_gen_1)
tstahg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_heat_gen_1)
tstahr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_heat_run_1)
tstahr4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_heat_run_4)
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
tstagr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_glace_run_1)
tsthas:
	(cd $(BUILD_PATH)/tests && $(HYODA) $(CTEST) -V -R nabla_arcane_schrodinger_run_1)
tsthas4:
	(cd $(BUILD_PATH)/tests && $(HYODA) $(CTEST) -V -R nabla_arcane_schrodinger_run_4)
tsts:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_sethi_run_1)
tsths:
	(cd $(BUILD_PATH)/tests && $(HYODA) $(CTEST) -V -R nabla_arcane_sethi_run_1)
tsts4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_sethi_run_4)

tstapr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_pDDFV_run_1)
tstmh:
	(cd $(BUILD_PATH)/tests && $(HYODA) $(CTEST) -V -R nabla_arcane_pDDFV_run_1)
tstm4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_pDDFV_run_4)


##############
# CUDA tests #
##############
tstugg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_gram_gen_1)
tstumg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_mhydro_gen_1)
tstulg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_lulesh_gen_1)
tstulr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_lulesh_run_1)
tstukg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_nvknl_gen_1)
tstukr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_nvknl_run_1)
tstudg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_darcy_gen_1)
tstudr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_cuda_darcy_run_1)


############
# CLEANING #
############
cln:;(cd $(BUILD_PATH) && make clean)
clean:cln

###########
# PHONIES #
###########
.PHONY: all cfg config bin tst cln clean
