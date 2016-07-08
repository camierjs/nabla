include(${NABLA_SOURCE_DIR}/cmake/CMakeTPL.txt)

info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_MAGENTA}LIBRARY${VT100_RESET}: ${VT100_BOLD}${VT100_FG_MAGENTA}NablaAleph${VT100_RESET}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(ALEPH_SOURCES
    AlephKernel.cc   AlephMatrix.cc
    AlephVector.cc   AlephParams.cc
    AlephTopology.cc AlephOrdering.cc
    AlephIndexing.cc AlephFactory.cc)


# Looking for PETSc
find_package(PETSc)
if(PETSc_FOUND)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/petsc)
endif()


# Looking for HYPRE
if(HYPRE_FOUND)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hypre)
  set(ALEPH_HYPRE_LIB ${HYPRE_LIB} PARENT_SCOPE)
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


add_library(Aleph ${ALEPH_SOURCES})

set(LIB_ALEPH ${CMAKE_CURRENT_BINARY_DIR}/libAleph.a PARENT_SCOPE)
set(LIB_ALEPH_HYPRE ${LIB_ALEPH_HYPRE} PARENT_SCOPE)
