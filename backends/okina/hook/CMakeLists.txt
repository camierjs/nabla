info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: OKINA/${VT100_BOLD}${VT100_FG_YELLOW}HOOK${VT100_RESET}")
set(BACKEND_OKINA_HOOK_C
  args.c hook.c forall.c header.c reduce.c
  params.c data.c variables.c tokens.c main.c mesh.c source.c)

add_library(NablaOkinaHook ${BACKEND_OKINA_HOOK_C})

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaOkinaHook NablaOkina)

