info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: CUDA/${VT100_BOLD}${VT100_FG_YELLOW}HOOK${VT100_RESET}")
set(BACKEND_CUDA_HOOKS_C
  nCudaHook.c
  nCudaHookArgs.c
  nCudaHookMain.c
  nCudaHookVars.c
  nCudaHookMesh.c
  nCudaHookData.c
  nCudaHookHeader.c
  nCudaHookTokens.c
  nCudaHookSource.c
  nCudaHookGather.c
  nCudaHookForAll.c
  nCudaHookScatter.c
  nCudaHookParameters.c
  nCudaHookReduction.c

  nCudaHookDiffraction.c
  nCudaHookFunction.c
  nCudaHookItem.c
  )

add_library(NablaCudaHook ${BACKEND_CUDA_HOOKS_C})

target_link_libraries(NablaCudaHook NablaCudaDump)

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaCudaHook NablaCuda)

