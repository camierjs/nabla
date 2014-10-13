export TGT=lulesh

#export SIMD=std
#export SIMD=sse
#export SIMD=avx
#export SIMD=mic

export COLORS=soa

export LOG=-v $(TGT).log
#export ADDITIONAL_NABLA_FILES = dbgLulesh.n


MESHs = 32 48 64 96 #8 16 24 32 40 48 56 64 72 80 88 96


all:
	for msh in $(MESHs);do\
		$(MAKE) --no-print-directory $$msh;\
	done;


# Generate command to launch
# 32 60 80 100 120 140 160 180 200 220 240
omp:
	@tput reset
	@for msh in $(MESHs);do\
		echo echo -ne ./lulesh_$$msh \&\&\\;\
		echo -e sync \&\& ./lulesh_$$msh \&\&\\;\
	done;




define TARGET_TEMPLATE
$(1): export MESH=$(1)
$(1): export X_EDGE_ELEMS = $(1)
$(1): export Y_EDGE_ELEMS = $$(shell echo $(1)-1|bc -l)
$(1): export Z_EDGE_ELEMS = $$(shell echo $(1)-1|bc -l)
$(1): export X_EDGE_TICK = $$(shell echo "scale=6;($(1)-1)/1000"|bc -l)
$(1): export Y_EDGE_TICK = $$(shell echo "scale=6;($(1))/1000"|bc -l)
$(1): export Z_EDGE_TICK = $$(shell echo "scale=6;($(1))/1000"|bc -l)
$(1): export LENGTH = $$(shell echo "scale=6;($(1)*($(1)-1))/1000"|bc -l)
$(1): export MESH_FLAGS = -DX_EDGE_ELEMS=$$(X_EDGE_ELEMS) -DY_EDGE_ELEMS=$$(Y_EDGE_ELEMS) -DZ_EDGE_ELEMS=$$(Z_EDGE_ELEMS) -DX_EDGE_TICK=$$(X_EDGE_TICK) -DY_EDGE_TICK=$$(Y_EDGE_TICK) -DZ_EDGE_TICK=$$(Z_EDGE_TICK) -DLENGTH=$$(LENGTH)
$(1):;@echo MESH_FLAGS = $$(MESH_FLAGS)
	$(MAKE) --no-print-directory -f ../bin/Makefile -kB all cuda
endef
$(foreach msh, $(MESHs),\
		$(eval $(call TARGET_TEMPLATE,$(msh))))


#include ../bin/Makefile
#include ../bin/Makefile.$(shell cea_os)