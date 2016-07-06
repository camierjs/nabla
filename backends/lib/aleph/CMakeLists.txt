include(${NABLA_SOURCE_DIR}/cmake/CMakeTPL.txt)

info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_MAGENTA}LIBRARY${VT100_RESET}: ${VT100_BOLD}${VT100_FG_MAGENTA}ALEPH${VT100_RESET}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(ALEPH_SOURCES AlephKernel.cc   AlephMatrix.cc
                  AlephVector.cc   AlephParams.cc
                  AlephTopology.cc AlephOrdering.cc
                  AlephIndexing.cc AlephFactory.cc
                  nablaAleph.c ${CMAKE_CURRENT_BINARY_DIR}/nablaAleph.S)
                
configure_file(nablaAleph.S ${CMAKE_CURRENT_BINARY_DIR}/nablaAleph.S)
                
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/nablaAleph.S 
  PROPERTY OBJECT_DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.*)


# Looking for PETSc
find_package(PETSc)
if(PETSc_FOUND)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/petsc)
endif()

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


configure_file(Aleph.h ${CMAKE_CURRENT_BINARY_DIR}/Aleph.h)
configure_file(AlephTypesSolver.h ${CMAKE_CURRENT_BINARY_DIR}/AlephTypesSolver.h)

add_library(NablaLibAleph ${ALEPH_SOURCES})

set(ALEPH_LIB ${CMAKE_CURRENT_BINARY_DIR}/libNablaLibAleph.a PARENT_SCOPE)
