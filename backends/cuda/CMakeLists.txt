info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND CUDA #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")


# BACKEND CUDA
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}CUDA${VT100_RESET}")

set(BACKEND_CUDA_C nCuda.c)

add_library(NablaCuda ${BACKEND_CUDA_C})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dump)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hook)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/cuSolver)

target_link_libraries(NablaCuda
  NablaCudaHook NablaCuSolver)

add_dependencies(NablaCuda libNabla)
