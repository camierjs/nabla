info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: KOKKOS/${VT100_BOLD}${VT100_FG_YELLOW}HOOK${VT100_RESET}")
set(BACKEND_KOKKOS_HOOK_C
    hook.c args.c
    reduce.c variables.c header.c 
    data.c main.c source.c
    forall.c mesh.c params.c tokens.c)

add_library(NablaKokkosHook ${BACKEND_KOKKOS_HOOK_C})

target_link_libraries(NablaKokkosHook NablaLibDump)

add_dependencies(NablaKokkosHook NablaKokkos)

