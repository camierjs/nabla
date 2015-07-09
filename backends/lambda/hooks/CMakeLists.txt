info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: LAMBDA/${VT100_BOLD}${VT100_FG_YELLOW}HOOKS${VT100_RESET}")
set(BACKEND_LAMBDA_HOOKS_C
  nLambdaHookArguments.c
  nLambdaHookDiffraction.c
  nLambdaHookGather.c
  nLambdaHookReductions.c
  nLambdaHookTokens.c
  nLambdaHook.c
  nLambdaHookForLoops.c
  nLambdaHookParameters.c
  nLambdaHookScatter.c)

add_library(NablaLambdaHooks ${BACKEND_LAMBDA_HOOKS_C})

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaLambdaHooks Nabla)

