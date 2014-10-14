export CC=/usr/local/opendev1/gcc/gcc/4.9.0/bin/gcc
export CXX=/usr/local/opendev1/gcc/gcc/4.9.0/bin/g++

CMAKE_PATH = /usr/local/opendev1/gcc/cmake/3.0.0
CMAKE = $(CMAKE_PATH)/bin/cmake
CTEST = $(CMAKE_PATH)/bin/ctest

BUILD_PATH = /tmp/$(USER)/nabla

all:
	(cd $(BUILD_PATH) && make)

tst:
	(cd $(BUILD_PATH)/tests && ctest)
tstg:
	(cd $(BUILD_PATH)/tests && ctest -R gen)
tstga2:
	(cd $(BUILD_PATH)/tests && ctest -V -R nabla_okina_lulesh_mic_gen_1)
tstra2:
	(cd $(BUILD_PATH)/tests && ctest -V -R nabla_okina_lulesh_mic_run_1_avx2)
tstro:
	(cd $(BUILD_PATH)/tests && ctest -V -R run_omp)
tstrc:
	(cd $(BUILD_PATH)/tests && ctest -V -R run_cilk)
tstr:
	(cd $(BUILD_PATH)/tests && ctest -R run)
tst4:
	(cd $(BUILD_PATH)/tests && ctest -j 4)
tstv:
	(cd $(BUILD_PATH)/tests && ctest -V)

build:
	(mkdir -p $(BUILD_PATH) && cd $(BUILD_PATH) && $(CMAKE) ~nabla/root)

cln:
	(cd $(BUILD_PATH) && make clean)
#\rm -rf $(BUILD_PATH)/*

