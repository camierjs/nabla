MFEM_PATH = /home/camierjs/mfem-3.1

CXXFLAGS = -g -Wall -std=gnu++11 -O3
LD_FLAGS = -L$(MFEM_PATH) -lmfem -lrt
ORDER = 3
TGT = ex1

PF_FLAGS = record
all:
	g++ $(CXXFLAGS) $(TGT).c -o $(TGT) $(LD_FLAGS)

go:
	./$(TGT) --visualization --order $(ORDER) --mesh ./star.mesh 

ex1:
	g++ -E -P -I$(MFEM_PATH) $(MFEM_PATH)/examples/ex1.cpp -o ex1.c

perf:
	perf $(PF_FLAGS) ./$(TGT) --order $(ORDER) --mesh ./star.mesh
	perf report

vis:
	glvis -m star.mesh -g sol.gf

cln:
	\rm -f perf.* refined.mesh sol.gf $(TGT)

