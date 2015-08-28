# Nabla Frontend
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}FRONTEND${VT100_RESET}")
set(FRONTEND_C
  nablaAstDfs.c
  nablaAstNode.c
  nablaAstTree.c
  nablaDebug.c)

add_library(NablaFrontend ${FRONTEND_C})
add_dependencies(NablaFrontend libNabla)