include(${NABLA_SOURCE_DIR}/CMakeTPL.txt)

info("\t${VT100_FG_YELLOW}Aleph::Hypre${VT100_RESET}")

include_directories(${MPI_PATH}/include)
include_directories(${HYPRE_INC}/include)
include_directories(${OKINA_SOURCE_DIR})
add_library(NablaOkinaAlephHypre IAlephHypre.cc)
