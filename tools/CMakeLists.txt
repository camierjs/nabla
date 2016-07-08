# Nabla Frontend
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}TOOLBOX${VT100_RESET}")
set(TOOLBOX_C files.c  ops.c  string.c  utf8.c sdup.c)

add_library(NablaToolbox ${TOOLBOX_C})
add_dependencies(NablaToolbox libNabla)
