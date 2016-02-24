GCC_VERSION = 4.9.2
NABLA_PATH = /tmp/$(USER)/nabla

TGT=ndspmhd
#LOG=-v ndspmhd.log
ADDITIONAL_NABLA_FILES	= 	initialise.n evolve.n step_leapfrog.n derivs.n \
									rates.n kernels.n \
								 	conservative2primitive.n iterate_density.n \
								 	write_header.n readwrite_infile.n \
								 	setup_random2D.n set_uniform_distribution.n\
								 	random.n check_setup.n inlines.n\
									density_sum.n eos.n\
									write_dump.n
#\
divBcorrect.n get_divB.n\
riemannsolver.n\
evwrite_mhd.n\
external_forces.n\
direct_sum_poisson.n get_curl.n 

ARCANE_ROOT_PATH = /usr/local/arcane/testing
ARCANE_AXL_EXE  = $(ARCANE_ROOT_PATH)/bin/axl2cc
#ARCANE_EXTRA_LIBS = $(shell PKG_CONFIG_PATH=/cea/S/temp/s3/camierjs/arcane/RedHat-6-x86_64 pkg-config --libs arcane)
ARCANE_LIBS = -L$(ARCANE_ROOT_PATH)/lib\
 -larcane_std\
 -larcane_mpi\
 -larcane_impl\
 -larcane_mesh\
 -larcane_core\
 -larcane_utils\
 -larcane_driverlib\
 $(ARCANE_EXTRA_LIBS)\
 -Wl,-rpath,$(ARCANE_ROOT_PATH)/lib
ARCANE_CFLAGS = $(shell PKG_CONFIG_PATH=$(ARCANE_ROOT_PATH)/lib/pkgconfig pkg-config --cflags arcane)
ARCANE_HYODA = $(ARCANE_ROOT_PATH)/bin/hyoda
# -larcane_mpithread\
 -larcane_thread\
 -larcane_aleph\
 -larcane_aleph_sloop\
 -larcane_aleph_hypre\
 -larcane_hyoda\
 -larcane_cea\


#########################
# THIRD PARTY LIBRARIES #
#########################
MPI_PATH = /usr/lib/openmpi


#############
# COMPILERS #
#############
#CXX=/usr/local/bin/g++
IFLAGS=-I$(MPI_PATH)/include
CFLAGS = -O2 -std=c++11 -Wall $(IFLAGS) $(ARCANE_CFLAGS) 
#LIBS=-Wl,-rpath,/usr/local/opendev1/gcc/gcc/$(GCC_VERSION)/lib64 \
#	/usr/local/sr/lib/libcea_user.a
LIBS=-L$(MPI_PATH)/lib -lmpi -Wl,-rpath,$(MPI_PATH)/lib


##########
# Target #
##########
$(TGT)Module.cc:$(TGT).n
	$(NABLA_PATH)/nabla $(LOG) --arcane --alone $(TGT) -i $(TGT).n $(ADDITIONAL_NABLA_FILES)
	$(ARCANE_AXL_EXE) -l c++ $(TGT).axl

$(TGT)Module.o: $(TGT)Module.cc
	$(CXX) -c $(CFLAGS) $(EXTRA_DEFINES) $(TGT)Module.cc -o $(TGT)Module.o

all:$(TGT)Module.cc $(TGT)Module.o
	$(CXX) -c $(CFLAGS) main.cc -o main.o
	$(CXX) -g -O2 -o $(TGT) $(TGT)Module.o main.o -Wl,-rpath,. $(LIBS) $(ARCANE_LIBS)

cln:
	-rm -f *.o *.a *.so *.co ${TGT} ${TGT}.axl ${TGT}_axl.h ${TGT}Module.* 
	-rm -f ${TGT}.E.* *.log *.dot *.svg main.cc mcuda.config
	-rm -rf output* signal* fatal* logs *.o *.a *.so *.co ${TGT} ${TGT}Module*
	-rm -rf ${TGT}.E.* *.log *.dot *.svg main.cc *.config
n:
	n $(ADDITIONAL_NABLA_FILES)
