include(${NABLA_SOURCE_DIR}/CMake/CMakeTPL.txt)

info("\t${VT100_FG_YELLOW}Aleph::Trilinos${VT100_RESET}")

include_directories(${MPI_INCLUDE_PATH})
include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${OKINA_SOURCE_DIR})
add_library(NablaOkinaAlephTrilinos IAlephTrilinos.cc)
