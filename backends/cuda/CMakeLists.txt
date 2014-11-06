#info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND CUDA #${VT100_RESET}")
#info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")


# BACKEND CUDA
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}CUDA${VT100_RESET}")

set(BACKEND_CUDA_C nccCudaMain.c nccCuda.c 
                      nccCudaEnumerates.c nccCudaVariables.c
                      nccCudaMesh.c nccCudaFunction.c
                      nccCudaJob.c nccCudaDump.S nccCudaSimd.c)

# On spécifie en plus que nccCudaDump.S va utiliser incbin pour se compiler
set_property(SOURCE nccCudaDump.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/outputs/*.*)

configure_file(outputs/real3.h ${CMAKE_CURRENT_BINARY_DIR}/real3.h)
configure_file(outputs/debug.h ${CMAKE_CURRENT_BINARY_DIR}/debug.h)
configure_file(outputs/error.h ${CMAKE_CURRENT_BINARY_DIR}/error.h)
configure_file(outputs/extra.h ${CMAKE_CURRENT_BINARY_DIR}/extra.h)


add_library(NablaCuda ${BACKEND_CUDA_C})
add_dependencies(NablaCuda Nabla)
