info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND KOKKOS #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

##################
# Kokkos BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}Kokkos${VT100_RESET}")

set(BACKEND_KOKKOS_C kokkos.c
    hook.c
    main.c
    vars.c
    mesh.c
    data.c
    header.c
    tokens.c
    source.c
    gather.c
    forall.c
    scatter.c
    params.c
    reduce.c)

add_library(NablaKokkos ${BACKEND_KOKKOS_C})
add_dependencies(NablaKokkos libNabla)
