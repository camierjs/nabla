GCC_VERSION = 4.8.1
NABLA_PATH = .


TGT=glace
LOG= # -t #-v glace.log
ADDITIONAL_NABLA_FILES = glaceGeomQuads.n glaceGeomTriangles.n glaceGeomHexa.n glaceBoundaries.n


ARCANE_PATH = /usr/local/arcane/gcc47-mpi/1.22.4
ARCANE_AXL_EXE  = $(ARCANE_PATH)/bin/axl2cc
ARCANE_LIBS = -L$(ARCANE_PATH)/lib\
 -larcane_std -larcane_mpithread -larcane_mpi -larcane_thread\
 -larcane_impl -larcane_mesh -larcane_core -larcane_utils\
 -larcane_driverlib -larcane_cea $(ARCANE_EXTRA_LIBS)\
 -Wl,-rpath,$(ARCANE_PATH)/lib
ARCANE_CFLAGS = -std=c++11 -I$(ARCANE_PATH)/include
ARCANE_HYODA = $(ARCANE_PATH)/hyoda


#########################
# THIRD PARTY LIBRARIES #
#########################
MPI_PATH = /usr/local/openmpi-1.5.3/gnu


#############
# COMPILERS #
#############
CXX=/usr/local/opendev1/gcc/gcc/$(GCC_VERSION)/bin/g++
IFLAGS=-I$(MPI_PATH)/include
CFLAGS = -O2 -std=c++11 -Wall $(IFLAGS) $(ARCANE_CFLAGS) 
LIBS = -Wl,-rpath,/usr/local/opendev1/gcc/gcc/$(GCC_VERSION)/lib64 \
	/usr/local/sr/lib/libcea_user.a\
	-L$(MPI_PATH)/lib -lmpi -Wl,-rpath,$(MPI_PATH)/lib


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


g1:$(TGT)
	ARCANE_PARALLEL_SERVICE=Mpi $(MPI_PATH)/bin/mpiexec -n 1 ./$(TGT) ./$(TGT).arc
g4:$(TGT)
	ARCANE_PARALLEL_SERVICE=Mpi $(MPI_PATH)/bin/mpiexec -n 4 ./$(TGT) ./$(TGT).arc
h1:$(TGT)
	ARCANE_PARALLEL_SERVICE=Mpi $(ARCANE_HYODA) $(MPI_PATH)/bin/mpiexec -n 1 $(shell pwd)/$(TGT) $(shell pwd)/$(TGT).arc

cln:
	-rm -f *.o *.a *.so *.co ${TGT} ${TGT}.axl ${TGT}_axl.h ${TGT}Module.* 
	-rm -f ${TGT}.E.* *.log *.dot *.svg main.cc mcuda.config
	-rm -rf output* signal* fatal* logs *.o *.a *.so *.co ${TGT} ${TGT}Module*
	-rm -rf ${TGT}.E.* *.log *.dot *.svg main.cc *.config
