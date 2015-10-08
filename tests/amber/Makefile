TGT=amber

MESH=4
SIMD=std
PARALLEL=seq
LOG=-t -v $(TGT).log
ADDITIONAL_NABLA_FILES=analytics.n ini.n aleph.n 

include ./Makefile.nabla.lambda 

gp:
	~/bin/gnuplot schema1.gp > schema1.svg
