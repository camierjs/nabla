CXXFLAGS = -g -Wall -std=gnu++11 -O3
LD_FLAGS = -L. -lmfem -lrt
ORDER = 6

PF_FLAGS = record
all:
	g++ $(CXXFLAGS) mfmx1.c -o mfmx1 $(LD_FLAGS)

go:
	./mfmx1 --visualization --order $(ORDER) --mesh ./star.mesh 

perf:
	perf $(PF_FLAGS) ./mfmx1 --visualization --order $(ORDER) --mesh ./star.mesh
	perf report

vis:
	./glvis -m star.mesh -g sol.gf

cln:
	\rm -f perf.* refined.mesh sol.gf mfmx1

