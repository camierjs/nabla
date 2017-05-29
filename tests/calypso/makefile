TGT=calypso

NABLA = /usr/local/nabla/bin/nabla
NOPTS = --lambda $(TGT)
#NOPTS = -v $(TGT).log --lambda $(TGT)
#NOPTS = -v $(TGT).log --arcane --alone $(TGT)
#FLAGS = -std=c++11 -g -O2 -Wall -DNO_SSE2 -mno-avx
FLAGS = -std=c++11 -g -O2 -Wall -mavx

LIBS = /usr/local/nabla/lib/libNablaOrgOpt.a \
		 /usr/local/nabla/lib/libNablaGLVis.a \
		 /usr/local/mfem/lib/libmfem.a \
		-lnetcdf
all: $(TGT)

NFs = calypso.n maths.n ntcdf.n \
		depth.n height.n velU.n velV.n

tst:
	./calypso -option_fill=false -option_max_iterations=8192 && ./calypso

gom:
#	tput reset && ./calypso -option_fill true -option_debug true -option_max_iterations=9|more
	tput reset && ./calypso -option_max_iterations=1025|more

gomf:
	tput reset && ./calypso -option_fill false -option_debug true -option_max_iterations=9|more

go:
	tput reset && ./calypso -option_fill true -option_debug true -option_max_iterations=9

VOPTS = -vis 7 -option_fill=false -option_max_iterations=16384 -option_stoptime=100
vis:
	./calypso $(VOPTS) -NX=64 -NY=48 -X_EDGE_ELEMS=66 -Y_EDGE_ELEMS=50 

monai:
	./calypso -vis 7 -option_fill=false -option_max_iterations=16384 -option_stoptime=100 -NX=393 -NY=244 -X_EDGE_ELEMS=395 -Y_EDGE_ELEMS=246 

$(TGT).cc: makefile $(NFs)
	$(NABLA) $(NOPTS) -i $(NFs)

$(TGT):$(TGT).cc
	$(CXX) $(FLAGS) -o $(TGT) $(TGT).cc $(LIBS)

cln:
	\rm -rf $(TGT) $(TGT).cc $(TGT).h $(TGT).log $(TGT)*.dot perf.*
