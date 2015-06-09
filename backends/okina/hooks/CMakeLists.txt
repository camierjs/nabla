info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: OKINA/${VT100_BOLD}${VT100_FG_YELLOW}HOOKS${VT100_RESET}")
set(BACKEND_OKINA_HOOKS_C
  nOkinaHooks.c
  nOkinaHookDiffraction.c
  nOkinaHookEnumerates.c
  nOkinaHookItem.c
  nOkinaHookReduction.c
  nOkinaHookFunction.c
  nOkinaHookParams.c
  nOkinaHookVariables.c
  nOkinaHookTokens.c nOkinaHookGather.c nOkinaHookScatter.c)

add_library(NablaOkinaHooks ${BACKEND_OKINA_HOOKS_C})

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaOkinaHooks Nabla)

