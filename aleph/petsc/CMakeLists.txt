include(${NABLA_SOURCE_DIR}/cmake/CMakeTPL.txt)

info("\t${VT100_FG_YELLOW}Aleph::PETSc${VT100_RESET}")

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PETSc_PATH}/include)
include_directories(${OKINA_SOURCE_DIR})
add_library(NablaOkinaAlephPETSc IAlephPETSc.cc)
