nabla = /tmp/nabla/nabla/nabla
CFLAGS = -std=c++11 -g -O2 -Wall -DNO_SSE2 -mno-avx
LIB_ORGOPT = /tmp/nabla/orgopt/libNablaOrgOpt.a
#LOG = -v pnnnt.log

lambda: pnnnt.cc
	g++ $(CFLAGS) -o pnnnt pnnnt.cc $(LIB_ORGOPT)

tst:
	tput reset && ./pnnnt --org pnnnt.org 

%.cc:%.n
	$(nabla) $(LOG) --lambda $* -i $*.n

cln:
	\rm -f *.axl *.cc *.h *.config *.log
