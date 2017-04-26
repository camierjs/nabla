TGT=calypso

NABLA = /usr/local/nabla/bin/nabla
NOPTS = -v $(TGT).log --lambda $(TGT)
#NOPTS = -v $(TGT).log --arcane --alone $(TGT)
FLAGS = -std=c++11 -g -O2 -Wall -DNO_SSE2 -mno-avx

LIBS = /usr/local/nabla/lib/libNablaOrgOpt.a \
		 /usr/local/nabla/lib/libNablaGLVis.a \
		 /usr/local/mfem/lib/libmfem.a \
		-lnetcdf
all:$(TGT)

go:
	tput reset && ./calypso

$(TGT).cc:$(TGT).n netcdf.n dump.n
	$(NABLA) $(NOPTS) -i $(TGT).n netcdf.n dump.n analytics.n

$(TGT):$(TGT).cc
	$(CXX) $(FLAGS) -o $(TGT) $(TGT).cc $(LIBS)

cln:
	\rm -rf $(TGT) $(TGT).cc $(TGT).h $(TGT).log
