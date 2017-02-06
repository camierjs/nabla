# source /opt/intel/parallel_studio_xe_2017.0.014/bin/psxevars.sh
# rm fem/lininteg.o fem/linearform.o fem/H1_QuadrilateralElement.o fem/Poly_1D.o fem/bilininteg.o fem/okina.o
# touch fem/lininteg.cpp fem/linearform.cpp fem/H1_QuadrilateralElement.cpp fem/Poly_1D.cpp fem/bilininteg.cpp

CXX = g++ #icpc

MFEM_PATH = /usr/local/mfem

CXXFLAGS = -Wall -g -std=gnu++11 -O3 -march=core-avx2 -I$(MFEM_PATH)/include
LD_FLAGS = -L$(MFEM_PATH)/lib -lmfem #-lrt

TGT = ex1

ORDER = 4
LEVEL = 6

MESH = star
#MESH = quad
#MFEM_DBG = -dbg
ASSEMBLE = --assemble_only
#ASSEMBLE = --linear_only
OPTIONS = -no-vis --order $(ORDER) --mesh ./$(MESH).mesh --level $(LEVEL) $(ASSEMBLE) $(MFEM_DBG)


PF_FLAGS = record
all:
	$(CXX) $(CXXFLAGS) $(TGT).c -o $(TGT) $(LD_FLAGS)

go:
	./$(TGT) $(OPTIONS)
goTst:
	./ex1 -no-vis --order 4 --mesh ./star.mesh --level 2

godb:
	gdb --args ./$(TGT) $(OPTIONS)
godbo:
	MFEM_OKINA=1 gdb --args ./$(TGT) $(OPTIONS)

goo:
	MFEM_OKINA=1 ./$(TGT) $(OPTIONS)

tst:
	\rm -f sol.* && sync
	make goTst > /dev/null
	cp sol.gf sol.ref.gf && sync 
	MFEM_OKINA=1 make goTst > /dev/null
	diff --brief -s sol.gf sol.ref.gf

#ex1:
#	g++ -E -P -I$(MFEM_PATH) $(MFEM_PATH)/examples/ex1.cpp -o ex1.c

perf:
	perf $(PF_FLAGS) ./$(TGT) $(OPTIONS) > /dev/null
	perf report

perfo:
	MFEM_OKINA=1 perf $(PF_FLAGS) ./$(TGT) $(OPTIONS) > /dev/null
	perf report

vis:
	glvis -m refined.mesh -g sol.gf
#	glvis -m $(MESH).mesh -g sol.gf

diff:
	diff -s sol.gf sol.ref.gf

cln:
	\rm -f perf.* refined.mesh sol.* $(TGT)

