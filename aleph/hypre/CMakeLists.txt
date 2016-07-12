include(${NABLA_SOURCE_DIR}/cmake/CMakeTPL.txt)

info("\t${VT100_FG_YELLOW}Aleph::HYPRE${VT100_RESET}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wliteral-suffix")

#include_directories(${MPI_INCLUDE_PATH})
include_directories(${HYPRE_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(AlephHYPRE IAlephHypre.cc)

add_dependencies(AlephHYPRE Aleph)

#set(LIB_ALEPH_HYPRE ${CMAKE_CURRENT_BINARY_DIR}/libAlephHYPRE.a PARENT_SCOPE)

install(TARGETS AlephHYPRE DESTINATION lib)
