info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: LAMBDA/${VT100_BOLD}${VT100_FG_YELLOW}PARALLEL${VT100_RESET}")
set(BACKEND_LAMBDA_PARALLEL_C
  nLambdaParallelCilk.c
  nLambdaParallelOpenMP.c
  nLambdaParallelVoid.c)

add_library(NablaLambdaParallel ${BACKEND_LAMBDA_PARALLEL_C})

#add_dependencies(NablaLambdaSimd Nabla)
