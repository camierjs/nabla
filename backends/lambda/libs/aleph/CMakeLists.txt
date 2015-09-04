include(${NABLA_SOURCE_DIR}/CMake/CMakeTPL.txt)

info("\t${VT100_FG_YELLOW}Aleph${VT100_RESET}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(ALEPH_SOURCES AlephKernel.cc   AlephMatrix.cc
                  AlephVector.cc   AlephParams.cc
                  AlephTopology.cc AlephOrdering.cc
                  AlephIndexing.cc AlephFactory.cc
                  nLambdaAleph.c ${CMAKE_CURRENT_BINARY_DIR}/nLambdaAleph.S)
                
configure_file(nLambdaAleph.S ${CMAKE_CURRENT_BINARY_DIR}/nLambdaAleph.S)

# Looking for PETSc
find_package(PETSc)
if(PETSc_FOUND)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/petsc)
endif()

# Looking for TRILINOS
find_path(Trilinos_CONFIG_PATH TrilinosConfig.cmake ${Trilinos_ROOT_PATH}/lib/cmake/Trilinos)
if (${Trilinos_CONFIG_PATH})
  include(${Trilinos_ROOT_PATH}/lib/cmake/Trilinos/TrilinosConfig.cmake)
  if(Trilinos_VERSION)
    # If we have it, add Kappa library
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/trilinos)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kappa)
  endif()
endif()


configure_file(Aleph.h ${CMAKE_CURRENT_BINARY_DIR}/Aleph.h)
configure_file(AlephTypesSolver.h ${CMAKE_CURRENT_BINARY_DIR}/AlephTypesSolver.h)

add_library(NablaLambdaAleph ${ALEPH_SOURCES})

add_dependencies(NablaLambdaAleph NablaLambdaHook)
