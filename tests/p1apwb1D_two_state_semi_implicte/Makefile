TGT=schema1
  
MESH=100
SIMD=std
PARALLEL=seq
LOG=-t #-v $(TGT).log
ADDITIONAL_NABLA_FILES=analytics.n  node.n  ini.n   norme_1.n  Bessel1.n
   #derives.n  #debug.n norme_1.n

include ./Makefile.nabla.okina 

gp:
	~/bin/gnuplot schema1.gp > schema1.svg