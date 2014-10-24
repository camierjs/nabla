include(${NABLA_SOURCE_DIR}/CMakeTPL.txt)

info("\t${VT100_FG_YELLOW}Aleph::Kappa${VT100_RESET}")

include_directories(${MPI_PATH}/include)
include_directories(${SLOOP_INC})
include_directories(${OKINA_SOURCE_DIR})
add_library(NablaOkinaAlephKappa AlephKappa.cc)
