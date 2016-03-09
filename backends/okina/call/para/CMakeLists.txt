info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: OKINA/${VT100_BOLD}${VT100_FG_YELLOW}PARALLEL${VT100_RESET}")
set(BACKEND_OKINA_PARALLEL_C nOkinaParallelCilk.c  nOkinaParallelOpenMP.c  nOkinaParallelVoid.c)

add_library(NablaOkinaCallPara ${BACKEND_OKINA_PARALLEL_C})

add_dependencies(NablaOkinaCallPara NablaOkina)
