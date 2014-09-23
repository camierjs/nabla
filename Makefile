export CC=/usr/local/opendev1/gcc/gcc/4.9.0/bin/gcc
export CXX=/usr/local/opendev1/gcc/gcc/4.9.0/bin/g++

CMAKE_PATH = /usr/local/opendev1/gcc/cmake/3.0.0
CMAKE = $(CMAKE_PATH)/bin/cmake
CTEST = $(CMAKE_PATH)/bin/ctest

BUILD_PATH = /tmp/$(USER)/nabla


all:build
	(cd $(BUILD_PATH) && make)

tst:
	(cd $(BUILD_PATH)/tests && ctest)
tstv:
	(cd $(BUILD_PATH)/tests && ctest -V)

build:
	mkdir -p $(BUILD_PATH) && cd $(BUILD_PATH) && $(CMAKE) ~nabla/nabla

cln:
	\rm -rf $(BUILD_PATH)

