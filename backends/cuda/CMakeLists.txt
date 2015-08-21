info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND CUDA #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")


# BACKEND CUDA
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}CUDA${VT100_RESET}")

set(BACKEND_CUDA_C
  nCuda.c 
  nCudaArgs.c 
  nCudaMain.c
  nCudaEnum.c
  nCudaVariables.c
  nCudaHeader.c
  nCudaDump.S
  nCudaSimd.c)

# On spécifie en plus que nccCudaDump.S va utiliser incbin pour se compiler
set_property(SOURCE nccCudaDump.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/dump/*.*)

configure_file(dump/real3.h ${CMAKE_CURRENT_BINARY_DIR}/real3.h)
configure_file(dump/debug.h ${CMAKE_CURRENT_BINARY_DIR}/debug.h)
configure_file(dump/error.h ${CMAKE_CURRENT_BINARY_DIR}/error.h)
configure_file(dump/extra.h ${CMAKE_CURRENT_BINARY_DIR}/extra.h)
configure_file(dump/cuMsh.h ${CMAKE_CURRENT_BINARY_DIR}/cuMsh.h)

add_library(NablaCuda ${BACKEND_CUDA_C})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hooks)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mesh)

target_link_libraries(NablaCuda
  NablaCudaHooks
  NablaCudaMesh)

add_dependencies(NablaCuda Nabla)
