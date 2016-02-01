#######################
# MacOS||Linux switch #
# gcc -dM -E -x c++ /dev/null | less
#######################
ifeq ($(shell uname),Darwin)
CMAKE_ROOT_PATH=/opt/local/bin
COMPILER_ROOT_PATH=/opt/local/bin
COMPILER_POSTFIX= #-mp-4.9
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
	(cd $(BUILD_PATH)/tests && $(CTEST) --schedule-random -j $(NUMBR_PROCS))
test:tst
tst1:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j 1)
tstn:
	(cd $(BUILD_PATH)/tests && $(CTEST) -N)
tstg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS) -R gen)
tstp1g:
	(cd $(BUILD_PATH)/tests && $(CTEST) -j $(NUMBR_PROCS) -R p1apwb1D)
tstp1gv:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R p1apwb1D_implicite)

###############
# OKINA tests #
###############
tstoua:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_upwindAP_run_1_std_seq)
tstou:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_upwind_run_1_std_seq)
tstolr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_run_1) # -E omp) #_avx2_seq)
tstolg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_lulesh_gen_1_std_seq)
tstoa:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_aecjs_gen_1)
tstopr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_okina_p1apwb1D_semi_implicite_run_1_std_seq)

################
# LAMBDA tests #
################
tstlpr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_p1apwb1D_implicite_run_1_std)
tstlhltr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_hlt_run_1)
tstlhltg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_hlt_gen_1)
tstl:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda)
tstllr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_lulesh_run_1_seq)
tstlllg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_llsh_gen_1_seq)
tstlllr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_llsh_run_1_seq)
tstllo:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_lulesh_run_1_omp)
tstlau:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_upwindAP_run_1_seq)
tstlhg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_heat_gen_1_seq)
tstlhr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_heat_run_1_seq)
tstlag:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_aleph1D_gen_1_seq)
tstlar:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_aleph1D_run_1_seq)
tstla2d:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_aleph2D_run)
tstla2dr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_lambda_aleph2D_run_1_seq)
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
tstahltr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_hlt_run_1)
tstahltg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_hlt_gen_1)
tstaglocig:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_gloci_gen_1)
tstaanyg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_anyItem_gen_1)
tstaanyr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_anyItem_run_1)
tstadrcg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_drc_gen_1)
tstashr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_shydro_run_1)
tstamr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_mhydro_run_1)
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
tstaalfg:
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
tstasr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_schrodinger_run_1)
tstas4:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_schrodinger_run_4)
tstalr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_lulesh_run_1)
tstallg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_llsh_gen_1)
tstalg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_lulesh_gen_1)
tstagr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_glace_run_1)
tstag2Dr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_glace2D_run_1)
tstaglcr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_glc_run_1)
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

tstaag:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_aecjs_gen_1)

tstapr:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_pDDFV_run_1)
tstapg:
	(cd $(BUILD_PATH)/tests && $(CTEST) -V -R nabla_arcane_pDDFV_gen_1)
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

#############
# 2014~20XY #
#############
grep2015:
	@grep -r 2014~2015 *
sed2015:
	@find . -type f -exec grep -l 2014~2016 {} \; |xargs  sed -i 's/2014\~2015/2014\~2016/g'

###########
# PHONIES #
###########
.PHONY: all cfg config bin tst cln clean



###########
# Speedup #
###########
PERF_PATH=/tmp/lulesh
MAKEFILE_DUMP=TGT=lulesh\\nMESH ?= $$msh\\nSIMD ?= $$vec\\nPARALLEL ?= omp\\nLOG= \\\# -t \\\#-v $(TGT).log\\nADDITIONAL_NABLA_FILES=luleshGeom.n\\ninclude /tmp/nabla/tests/Makefile.nabla.okina


SIMDs = std sse avx avx2
MESHs = 16 24 32 40 48 56 64 72 80 88 96
THREADs = 1 2 4 8 16 20 24 32 48 64
ECHO=/bin/echo

gen:
	mkdir -v -p $(PERF_PATH)
	@tput reset
	@echo Now launching generation
	@for vec in $(SIMDs);do\
		for msh in $(MESHs);do\
				$(ECHO) -e \\t$$vec\\t$$msh;\
				mkdir -v -p $(PERF_PATH)/$$vec/$$msh;\
				ln -fs $(shell pwd)/tests/lulesh/*.n $(PERF_PATH)/$$vec/$$msh;\
				$(ECHO) -e $(MAKEFILE_DUMP) > $(PERF_PATH)/$$vec/$$msh/Makefile;\
				(cd $(PERF_PATH)/$$vec/$$msh && make);\
			done;\
		done;\
	done

go:
	@tput reset
	@echo Now launching tests
	@for vec in $(SIMDs);do\
		for msh in $(MESHs);do\
			for thr in $(THREADs);do\
				$(ECHO) -e \\t$$vec\\t$$msh\\t$$thr;\
				KMP_AFFINITY=scatter,granularity=fine OMP_NUM_THREADS=$$thr LC_NUMERIC=en_US.UTF8 perf stat -r 4 -e instructions,cycles,task-clock,cpu-clock \
-o $(PERF_PATH)/$$vec/$$msh/lulesh"_"$$msh"_"omp"_"$$vec"_"$$thr.perf \
   $(PERF_PATH)/$$vec/$$msh/lulesh"_"$$msh"_"omp"_"$$vec \> /dev/null;\
			done;\
		done;\
	done

get:
	@tput reset
	@echo Now collecting results
	@for vec in $(SIMDs);do\
		OUTPUT_FILE=/tmp/nablaLulesh"_"$$vec"."org;\
		$(ECHO) -e \\tGenerating: $$OUTPUT_FILE;\
		$(ECHO) -n > $$OUTPUT_FILE;\
		for msh in $(MESHs);do\
			$(ECHO) -e \\t\\tMESH=$$msh;\
			VAL_ONE=`cat /tmp/lulesh/$$vec/$$msh/lulesh_$$msh"_"omp"_"$$vec"_"1.perf|grep elapsed|cut -d's' -f1|tr -d [:blank:]`;\
			$(ECHO) $$VAL_ONE;\
			$(ECHO) \* lulesh_$$msh"_"omp"_"$$vec >> $$OUTPUT_FILE &&\
			for thr in $(THREADs);do\
				$(ECHO) -e \\t\\t\\tTHREAD=$$thr;\
				(sync && sync &&\
					$(ECHO) -n \|$$vec\|$$thr\|$$msh\|$$VAL_ONE\| >> $$OUTPUT_FILE &&\
					cat /tmp/lulesh/$$vec/$$msh/lulesh_$$msh"_"omp"_"$$vec"_"$$thr.perf|grep elapsed|cut -d's' -f1|tr -d [:blank:]|tr -d \\n >> $$OUTPUT_FILE &&\
					$(ECHO) \| >> $$OUTPUT_FILE);\
			done;\
			$(ECHO) >> $$OUTPUT_FILE;\
		done;\
	done
