include(${NABLA_SOURCE_DIR}/CMake/CMakeTPL.txt)

info("\t${VT100_FG_YELLOW}Aleph${VT100_RESET}")

set(ALEPH_SOURCES AlephKernel.cc   AlephMatrix.cc
                  AlephVector.cc   AlephParams.cc
                  AlephTopology.cc AlephOrdering.cc
                  AlephIndexing.cc AlephFactory.cc)

set(OKINA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Looking for PETSc
find_package(PETSc)
if(PETSc_FOUND)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/petsc)
endif()

# Looking for TRILINOS
include(${Trilinos_ROOT_PATH}/lib/cmake/Trilinos/TrilinosConfig.cmake)
if(Trilinos_VERSION)
  # If we have it, add Kappa library
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/trilinos)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kappa)
endif()


add_library(NablaOkinaAleph ${ALEPH_SOURCES})
