###############################
# MacOS||Linux switch||CentOS #
# gcc -dM -E -x c++ /dev/null #
###############################
ifeq ($(shell uname),Darwin)
  CMAKE_ROOT_PATH=/opt/local/bin
  COMPILER_ROOT_PATH=/opt/local/bin
  COMPILER_POSTFIX= #-mp-4.9
endif
ifeq ($(shell uname),Linux)
  CMAKE_ROOT_PATH = /usr/bin
  COMPILER_ROOT_PATH = /usr/bin
endif
ifeq ($(shell [ -f /etc/redhat-release ] && cat /etc/redhat-release|cut -b 1-6),CentOS)
  CMAKE_ROOT_PATH = /usr/bin
  COMPILER_ROOT_PATH = /usr/local/gcc/bin
endif

####################
# COMPILER OPTIONS #
# gcc -dM -E - < /dev/null
####################
C_FLAGS = -std=c99
MAKEFLAGS = --no-print-directory
export CC  = $(COMPILER_ROOT_PATH)/gcc$(COMPILER_POSTFIX)
export CXX = $(COMPILER_ROOT_PATH)/g++$(COMPILER_POSTFIX)

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
tst:test
test:
	(cd $(BUILD_PATH)/tests && $(CTEST) --schedule-random -j $(NUMBR_PROCS))


####################
# BACKEND TEMPLATE #
####################
backends = arcane cuda kokkos lambda okina
define BACKEND_template =
$(1):
	(cd $(BUILD_PATH)/tests && $(CTEST) --schedule-random -j $(NUMBR_PROCS) -R $(1))
endef
$(foreach backend,$(backends),$(eval $(call BACKEND_template,$(backend))))


###################
# CTESTS TEMPLATE #
###################
tests = glace2D lulesh upwind #upwind deflex upwindAP lulesh darcy ndspmhd mhydro glace2D
#p1apwb1D_gosse heat aleph1D kripke darcy deflex llsh lulesh shydro sethi anyItem gad comd pDDFV
#$(shell cd tests && find . -maxdepth 1 -type d -name \
	[^.]*[^\\\(mesh\\\)]*[^\\\(gloci\\\)]*|\
		sed -e "s/\\.\\// /g"|tr "\\n" " ")
procs = 1 #1 4
types = gen run
simds = std #sse avx # avx2 mic warp
backends = lambda okina cuda #kokkos lambda arcane okina cuda 
parallels = seq smp #omp mpi smp #cilk
define CTEST_template =
nabla_$(1)_$(2)_$(3)_$(4)_$(5)_$(6):
	(tput reset && cd $(BUILD_PATH)/tests && \
		$(CTEST) -V -R nabla_$(1)_$(2)_$(3)_$(4)_$(5)_$(6))
endef
$(foreach type,$(types),\
	$(foreach backend,$(backends),\
		$(foreach test,$(tests),\
			$(foreach cpu,$(procs),\
				$(foreach simd,$(simds),\
					$(foreach parallel,$(parallels),\
						$(eval $(call CTEST_template,$(type),$(backend),$(test),$(cpu),$(simd),$(parallel)))))))))

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


#########################
# PERF TESTS GEN/GO/GET #
#########################
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
