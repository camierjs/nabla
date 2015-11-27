# Nabla MiddlEnd
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}MIDDLEND${VT100_RESET}")
set(MIDDLEND_C
  animate.c    entities.c   grammar.c  hlt.c    jobs.c  middlend.c  printf.c    types.c
  arguments.c  functions.c  header.c   items.c  libs.c  options.c   timeTree.c  variables.c
  globals.c init.c)

add_library(NablaMiddlend ${MIDDLEND_C})
add_dependencies(NablaMiddlend libNabla)