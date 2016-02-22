info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: KOKKOS/${VT100_BOLD}${VT100_FG_YELLOW}CALL${VT100_RESET}")
set(BACKEND_KOKKOS_CALL_C
    call.c
    gather.c scatter.c
    parallel/cilk.c parallel/openMP.c parallel/void.c)

add_library(NablaKokkosCall ${BACKEND_KOKKOS_CALL_C})

target_link_libraries(NablaKokkosCall)

add_dependencies(NablaKokkosCall NablaKokkos)

