info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: CUDA/${VT100_BOLD}${VT100_FG_YELLOW}DUMP${VT100_RESET}")

set(BACKEND_CUDA_DUMP_C
  nCudaDump.c
  ${CMAKE_CURRENT_BINARY_DIR}/nCudaDump.S)

# On spécifie en plus que nCudaDump.S va utiliser incbin pour se compiler
set_property(SOURCE
  ${CMAKE_CURRENT_BINARY_DIR}/nCudaDump.S 
  PROPERTY OBJECT_DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

configure_file(nCudaDump.S ${CMAKE_CURRENT_BINARY_DIR}/nCudaDump.S)

add_library(NablaCudaDump ${BACKEND_CUDA_DUMP_C})

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaCudaDump NablaCudaHook)

