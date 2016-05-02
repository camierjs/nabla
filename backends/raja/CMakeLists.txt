info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND RAJA #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")

##################
# Raja BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}RAJA${VT100_RESET}")

set(BACKEND_RAJA_C raja.c)

add_library(NablaRaja ${BACKEND_RAJA_C})
add_dependencies(NablaRaja libNabla)
