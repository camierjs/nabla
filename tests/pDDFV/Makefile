export TGT=pDDFV
export LOG=-t -v $(TGT).log
export EXTRA_DEFINES = -DALEPH_INDEX
export ADDITIONAL_NABLA_FILES = 	pDDFV_ini.n pDDFV_geom.n \
											pDDFV_aleph.n pDDFV_direct_aleph.n pDDFV_indirect_aleph.n\
										  	pDDFV_direct_primal.n pDDFV_indirect_primal.n\
											pDDFV_direct_dual.n pDDFV_indirect_dual.n \
											pDDFV_indirect_trial_average.n pDDFV_indirect_trial.n \
										  	pDDFV_analytics.n \
											pDDFV_diags.n \
											pDDFV_eieio.n\
										  	pDDFV_mesh.n pDDFV_dag.n\
										  	pDDFV_exit.n pDDFV_miax.n
# pDDFV_debug.n pDDFV_miax

#ssh:	ssh x-kadarka gmake --directory=~/cea/arcane/nabla/pDDFV -kB all

#include ../bin/Makefile
include ../bin/Makefile.$(shell cea_os)
#include ../bin/Makefile.$(shell cea_os).local

DOT_FLAGS = -O # -Ln2
GRAPHVIZ_PATH=/usr/bin
#local/opendev1/gcc/graphviz/2.30.1/bin

DOT = $(GRAPHVIZ_PATH)/dot
GVPACK = $(GRAPHVIZ_PATH)/gvpack
NEATO = $(GRAPHVIZ_PATH)/neato
CCOMPS = $(GRAPHVIZ_PATH)/ccomps

r:
	make g1|grep -e "\[relativeError\]" #\|\[stopIterations\]"
#r2:	make g2|grep -e "\[relativeError\]" #\|\[stopIterations\]"
#r4:	make g4|grep -e "\[relativeError\]" #\|\[stopIterations\]"

m:
	rm -f m_aleph_* && make r && make mm
mm:
	-rm -rf /tmp/*.mm
	for input in `ls m_aleph_matrix_A_*[!m]`;do\
		/usr/local/opendev1/gcc/mono/3.4.0/bin/mono --optimize=all --debug ~/usr/src/slp2mm/slp2mm.exe ./$$input;\
	done


pack: $(TGT)Module.cc
	/usr/bin/ccomps -Cx  pDDFV.time.dot |/usr/bin/dot|/usr/bin/gvpack -array_u1 |neato -n2 -Tsvg -o $(TGT).time.svg

#$(CCOMPS) -x pDDFV.time.dot 
#|$(DOT) pDDFV.time.dot |$(GVPACK) -array_u1 #|$(NEATO) -n1 -Tpng -O

dot: #$(TGT)Module.cc
	$(DOT) -Tsvg $(DOT_FLAGS) pDDFV.time.dot

samples:
	$(DOT) -Tsvg $(DOT_FLAGS) pDDFV.samples.dot


##################################################################
MESH_GENERATOR = "s/<mesh>.*<\/mesh>/<mesh>\n\
\t\t<meshgenerator>\n\
\t\t\t<cartesian>\n\
\t\t\t\t<nsd>8 1 1<\/nsd>\n\
\t\t\t\t<origine>0.0 0.0 0.0<\/origine>\n\
\t\t\t\t<lx nx=\"$(1)\" prx=\"1.0\">1.0<\/lx>\n\
\t\t\t\t<ly ny=\"$(1)\" pry=\"1.0\">1.0<\/ly>\n\
\t\t\t<\/cartesian>\n\
\t\t<\/meshgenerator>\n\
\t<\/mesh>/g"

##################################################################
MESHs = quad rdq sncq #quad rdq sncq sqr zzz
SIZEs = 10 20 40 80 #160 320 640 1280 2560
ISOTROPICs = isotropic anisotropic
#ISOTROPICs = anisotropic

CEA_HOME_TMP = $(shell cea_home -t)
out:
	@tput reset
	@echo Now copying executable
	mkdir -vp $(CEA_HOME_TMP)/$(TGT) && cp -t $(CEA_HOME_TMP)/$(TGT) ./$(TGT) ./$(TGT).config
	@echo Now making directories
	@for iso in $(ISOTROPICs);do\
		for mesh in $(MESHs);do\
			for size in $(SIZEs);do\
				dirName=$(CEA_HOME_TMP)/$(TGT)/$$iso/$$mesh/$$size;\
				echo -e \\tFocusing: $$dirName;\
				mkdir -vp $$dirName && sync && sync;\
				cp -t $$dirName ./$(TGT).arc;\
				ln -fvs -t $$dirName $(CEA_HOME_TMP)/$(TGT)/$(TGT) $(CEA_HOME_TMP)/$(TGT)/$(TGT).config;\
				sed -i $(call MESH_GENERATOR,$$size) $$dirName/$(TGT).arc;\
			done;\
		done;\
	done;

go:
	@tput reset
	@for iso in $(ISOTROPICs);do\
		for mesh in $(MESHs);do\
			for size in $(SIZEs);do\
				dirName=$(CEA_HOME_TMP)/$(TGT)/$$iso/$$mesh/$$size;\
				echo -e \\tLaunching: $$dirName;\
				(cd $$dirName && ARCANE_PARALLEL_SERVICE=Mpi /opt/mpi/bullxmpi/1.1.14.3/bin/mpiexec -n 1 ./$(TGT) ./$(TGT).arc > ./$(TGT).out);\
			done;\
		done;\
	done;
