info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: LAMBDA/${VT100_BOLD}${VT100_FG_YELLOW}HOOK${VT100_RESET}")
set(BACKEND_LAMBDA_HOOK_C
  nLambdaHook.c
  nLambdaHookMain.c
  nLambdaHookVars.c
  nLambdaHookMesh.c
  nLambdaHookData.c
  nLambdaHookHeader.c
  nLambdaHookTokens.c
  nLambdaHookSource.c
  nLambdaHookGather.c
  nLambdaHookScatter.c
  nLambdaHookForLoops.c
  nLambdaHookArguments.c
  nLambdaHookParameters.c
  nLambdaHookReductions.c
  nLambdaHookParallelCilk.c
  nLambdaHookParallelOpenMP.c
  nLambdaHookParallelVoid.c)

add_library(NablaLambdaHook ${BACKEND_LAMBDA_HOOK_C})

target_link_libraries(NablaLambdaHook NablaLambdaDump)

add_dependencies(NablaLambdaHook NablaLambda)

