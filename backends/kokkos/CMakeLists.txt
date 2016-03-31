info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND KOKKOS #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

##################
# Kokkos BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}Kokkos${VT100_RESET}")

set(BACKEND_KOKKOS_C kokkos.c hook.c vars.c main.c reduce.c forall.c ${CMAKE_CURRENT_BINARY_DIR}/kokkos.S)
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/kokkos.S 
             PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_dump.h)
configure_file(kokkos.S ${CMAKE_CURRENT_BINARY_DIR}/kokkos.S)

add_library(NablaKokkos ${BACKEND_KOKKOS_C})
add_dependencies(NablaKokkos libNabla)
