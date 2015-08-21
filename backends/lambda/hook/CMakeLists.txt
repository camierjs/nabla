info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: LAMBDA/${VT100_BOLD}${VT100_FG_YELLOW}HOOK${VT100_RESET}")
set(BACKEND_LAMBDA_HOOK_C
  nLambdaHookForwards.c
  nLambdaHookTypedef.c
  nLambdaHookDefines.c
  nLambdaHookVars.c
  nLambdaHookMesh.c
  nLambdaHookHeader.c
  nLambdaHookMain.c
  nLambdaHookSource.c
  nLambdaHookArguments.c
  nLambdaHookDiffraction.c
  nLambdaHookGather.c
  nLambdaHookReductions.c
  nLambdaHookTokens.c
  nLambdaHook.c
  nLambdaHookData.c
  nLambdaHookForLoops.c
  nLambdaHookParameters.c
  nLambdaHookScatter.c
  nLambdaHookParallelCilk.c
  nLambdaHookParallelOpenMP.c
  nLambdaHookParallelVoid.c)

add_library(NablaLambdaHook ${BACKEND_LAMBDA_HOOK_C})

target_link_libraries(NablaLambdaHook
  NablaLambdaDump)

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaLambdaHook NablaLambda)

