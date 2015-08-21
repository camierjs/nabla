info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: CUDA/${VT100_BOLD}${VT100_FG_YELLOW}HOOKS${VT100_RESET}")
set(BACKEND_CUDA_HOOKS_C
  nCudaHookDiffraction.c
  nCudaHookFunction.c
  nCudaHookItem.c
  nCudaHookReduction.c
  nCudaHookScatter.c
  nCudaHookVariables.c
  nCudaHookEnumerates.c
  nCudaHookGather.c
  nCudaHookParams.c
  nCudaHooks.c
  nCudaHookTokens.c)

add_library(NablaCudaHooks ${BACKEND_CUDA_HOOKS_C})

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaCudaHooks NablaCuda)

