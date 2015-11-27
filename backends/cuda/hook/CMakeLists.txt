info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: CUDA/${VT100_BOLD}${VT100_FG_YELLOW}HOOK${VT100_RESET}")
set(BACKEND_CUDA_HOOKS_C
  cuHook.c
  cuHookArgs.c
  cuHookMain.c
  cuHookVars.c
  cuHookMesh.c
  cuHookData.c
  cuHookHeader.c
  cuHookTokens.c
  cuHookSource.c
  cuHookGather.c
  cuHookForAll.c
  cuHookScatter.c
  cuHookParameters.c
  cuHookReduction.c

  cuHookDiffraction.c
  cuHookFunction.c
  cuHookItem.c
  )

add_library(NablaCudaHook ${BACKEND_CUDA_HOOKS_C})

target_link_libraries(NablaCudaHook NablaCudaDump)

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaCudaHook NablaCuda)

