# Nabla MiddlEnd
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}MIDDLEND${VT100_RESET}")
set(MIDDLEND_C
  nMiddle.c
  nMiddleHLT.c
  nMiddleJobs.c
  nMiddleTypes.c
  nMiddleItems.c
  nMiddlePrintf.c
  nMiddleOptions.c
  nMiddleEntities.c
  nMiddleTimeTree.c
  nMiddleVariables.c 
  nMiddleFunctions.c
  nMiddleLibraries.c)

add_library(NablaMiddlend ${MIDDLEND_C})
add_dependencies(NablaMiddlend Nabla)