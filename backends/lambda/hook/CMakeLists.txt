info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: LAMBDA/${VT100_BOLD}${VT100_FG_YELLOW}HOOK${VT100_RESET}")
set(BACKEND_LAMBDA_HOOK_C
  lambdaHookArgs.c lambdaHookGather.c lambdaHookParallelCilk.c lambdaHookReductions.c lambdaHookVars.c
  lambdaHook.c lambdaHookHeader.c lambdaHookParallelOpenMP.c lambdaHookScatter.c
  lambdaHookData.c lambdaHookMain.c lambdaHookParallelVoid.c lambdaHookSource.c
  lambdaHookForAll.c lambdaHookMesh.c lambdaHookParameters.c lambdaHookTokens.c
  )

add_library(NablaLambdaHook ${BACKEND_LAMBDA_HOOK_C})

target_link_libraries(NablaLambdaHook NablaLibDump)

if(HYPRE_FOUND)
    target_link_libraries(NablaLambdaHook NablaAleph)
endif()

add_dependencies(NablaLambdaHook NablaLambda)

