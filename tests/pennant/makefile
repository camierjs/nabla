nabla = /tmp/nabla/nabla/nabla
CFLAGS = -std=c++11 -g -O2 -Wall -DNO_SSE2 -mno-avx
ADDITIONAL_NABLA_FILES = pGeom.n
LIB_ORGOPT = /tmp/nabla/orgopt/libNablaOrgOpt.a
#LOG = -v pennant.log

lambda: pennant.cc
	g++ $(CFLAGS) -o pennant pennant.cc $(LIB_ORGOPT)

tst:
	tput reset && ./pennant --org pennant.org 

%.cc:%.n $(ADDITIONAL_NABLA_FILES)
	$(nabla) $(LOG) --lambda $* -i $*.n $(ADDITIONAL_NABLA_FILES)

arcane:
	$(nabla) --arcane --alone pennant -i pennant.n

cln:
	\rm -f *.axl *.cc *.h *.config *.log
