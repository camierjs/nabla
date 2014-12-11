include(${NABLA_SOURCE_DIR}/CMake/CMakeTPL.txt)

info("\t${VT100_FG_YELLOW}Aleph${VT100_RESET}")

set(ALEPH_SOURCES AlephKernel.cc   AlephMatrix.cc
                  AlephVector.cc   AlephParams.cc
                  AlephTopology.cc AlephOrdering.cc
                  AlephIndexing.cc AlephFactory.cc)

set(OKINA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kappa)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hypre)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sloop)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/petsc)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/trilinos)

add_library(NablaOkinaAleph ${ALEPH_SOURCES})
