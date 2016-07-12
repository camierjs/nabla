# Nabla Frontend
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}FRONTEND${VT100_RESET}")
set(FRONTEND_C dbg.c dfs.c astNode.c astTree.c rules.c)
add_library(NablaFrontend ${FRONTEND_C})
add_dependencies(NablaFrontend Nabla)

#install(TARGETS NablaFrontend DESTINATION lib)
