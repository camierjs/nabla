info("${VT100_FG_CYAN}ALEPH_${VT100_BOLD}CUDA${VT100_RESET} subdirectory")
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

INCLUDE_DIRECTORIES(${ARCANE_SRC_PATH})
INCLUDE_DIRECTORIES(${ARCANEBUILDROOT})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})
INCLUDE_DIRECTORIES(${ARCANE_CEA_SOURCE_PATH}/src)

ARCANE_INCLUDE_CONFIG(${ARCANE_CEA_SOURCE_PATH}/src ALEPH_CUDA aleph/cuda)

ADD_DEFINITIONS(-DARCANE_COMPONENT_arcane_aleph_cuda)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

set(ALEPH_CU ${CMAKE_CURRENT_SOURCE_DIR}/AlephCuda.cu)
set(ALEPH_CU_O CMakeFiles/arcane_aleph_cuda.dir/AlephCuda.cu.o)

add_custom_command(OUTPUT ${ALEPH_CU_O}
                   COMMAND ${CUDA_BIN_NVCC} 
                   ARGS -I${ARCANE_SRC_PATH} -I${CMAKE_BINARY_DIR} -I${ARCANE_CEA_SOURCE_PATH}/src -shared -arch=sm_21 --compiler-options -fPIC -DARCANE_NOEXCEPT_FOR_NVCC -c -o ${ALEPH_CU_O} ${ALEPH_CU}
	  				    DEPENDS ${ALEPH_CU}
                   COMMENT " ######## NVCC ########\n\t########################")

ADD_LIBRARY(arcane_aleph_cuda ${CMAKE_BINARY_DIR}/aleph/cuda/CMakeFiles/arcane_aleph_cuda.dir/AlephCuda.cu.o ${ARCANE_ALEPH_CUDA})

target_link_libraries(arcane_aleph_cuda ${ARCANE_BASE_LIBRARIES})
set_target_properties(arcane_aleph_cuda PROPERTIES INSTALL_RPATH_USE_LINK_PATH 1)
set_target_properties(arcane_aleph_cuda PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

INSTALL(TARGETS arcane_aleph_cuda DESTINATION lib)
