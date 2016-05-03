info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND RAJA #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")

##################
# Raja BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}RAJA${VT100_RESET}")

set(BACKEND_RAJA_C raja.c forall.c hook.c main.c reduce.c ${CMAKE_CURRENT_BINARY_DIR}/raja.S)

set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/raja.S 
             PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/raja_dump.h)
configure_file(raja.S ${CMAKE_CURRENT_BINARY_DIR}/raja.S)

add_library(NablaRaja ${BACKEND_RAJA_C})
add_dependencies(NablaRaja libNabla)
