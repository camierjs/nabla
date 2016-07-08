# Nabla MiddlEnd
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}MIDDLEND${VT100_RESET}")
set(MIDDLEND_C
  animate.c entities.c gram.c  hlt.c jobs.c printf.c types.c
  args.c functions.c header.c items.c libs.c options.c timeTree.c  vars.c
  globals.c middlend.c swirl.c)

add_library(NablaMiddlend ${MIDDLEND_C})

# Add math library for pow *sic*
target_link_libraries(NablaMiddlend m)

add_dependencies(NablaMiddlend libNabla)
