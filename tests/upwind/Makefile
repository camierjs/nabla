TGT=upwind
LOG=-t #-v $(TGT).log
MESH = 16
SIMD = std
PARALLEL = seq
NABLA = /tmp/nabla/nabla

export X_EDGE_ELEMS = $(MESH)
export LENGTH = $(shell echo 2.0)
export X_EDGE_TICK = $(shell echo "scale=6;1.125/$(MESH)"|bc -l)
export MESH_FLAGS = -DX_EDGE_ELEMS=$(X_EDGE_ELEMS) -DX_EDGE_TICK=$(X_EDGE_TICK) -DLENGTH=$(LENGTH)

all:$(TGT).cc
	/usr/bin/g++ -O3 -Wall -DWARP_BIT=0 -mno-avx -DNO_SSE2 -finline $(MESH_FLAGS) upwind.cc -o upwind

$(TGT).cc: $(TGT_FILE) $(NABLA)
	$(NABLA) $(LOG) --okina $(TGT) --$(SIMD) --$(PARALLEL) -i upwind.n analytics.n
