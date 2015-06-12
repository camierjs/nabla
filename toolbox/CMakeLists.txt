# Nabla Frontend
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}TOOLBOX${VT100_RESET}")
set(TOOLBOX_C
  nToolbox.c
  nToolsUtf8.c
  nToolsFile.c
  nToolsStrg.c
  nToolsOperators.c)

add_library(NablaToolbox ${TOOLBOX_C})
add_dependencies(NablaToolbox Nabla)