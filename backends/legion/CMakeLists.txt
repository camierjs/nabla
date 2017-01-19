info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND LEGION #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

##################
# Legion BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}Legion${VT100_RESET}")

set(BACKEND_LEGION_C legion.c hooks.c)

add_library(NablaLegion ${BACKEND_LEGION_C})
add_dependencies(NablaLegion Nabla)
