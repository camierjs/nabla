export TGT=mhydro
export SIMD=avx
#export ADDITIONAL_NABLA_FILES = dbgMHydroArcane.n
#export ADDITIONAL_NABLA_FILES = dbgMHydroPhi.n
#export LOG=-v mhydro.log
include ../bin/Makefile


mic:
	/usr/local/intel/Compiler/14.0/0.080/bin/icc -DWARP_BIT=3 -O1 -mmic -o mhydro mhydro.cc -L/usr/local/intel/Compiler/14.0/0.080/lib/mic -lcilkrts -Wl,-rpath,/usr/local/intel/Compiler/14.0/0.080/lib/mic
