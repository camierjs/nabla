info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: LAMBDA/${VT100_BOLD}${VT100_FG_YELLOW}PARA${VT100_RESET}")
set(BACKEND_LAMBDA_PARA_C
  nLambdaParallelCilk.c
  nLambdaParallelOpenMP.c
  nLambdaParallelVoid.c)

add_library(NablaLambdaPara ${BACKEND_LAMBDA_PARA_C})

add_dependencies(NablaLambdaPara NablaLambda)
