# Nabla MiddlEnd
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}MIDDLEND${VT100_RESET}")
set(MIDDLEND_C nablaMiddlend.c nablaItems.c nablaVariables.c 
               nablaOptions.c nablaModules.c nablaJobs.c nablaFunctions.c nablaLibraries.c)

add_library(NablaMiddlend ${MIDDLEND_C})
add_dependencies(NablaMiddlend Nabla)