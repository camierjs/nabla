info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}LIBRARY${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}CALL${VT100_RESET}")

set(BACKEND_LIB_CALL_C call.c gather.c scatter.c)

add_library(NablaLibCall ${BACKEND_LIB_CALL_C})
add_dependencies(NablaLibCall Nabla)
