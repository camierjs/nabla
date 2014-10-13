export TGT=gram
export LOG=-t -v $(TGT).log

export SIMD = std
export COLORS = aos
export PARALLEL = omp
#include ../bin/Makefile.$(shell cea_os).local
#include ../bin/Makefile.$(shell cea_os)
include ../bin/Makefile

png:
	/usr/local/opendev1/gcc/graphviz/2.30.1/bin/dot -Tpng gram.dot -o gram.png
