info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND KOKKOS #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

##################
# Kokkos BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}Kokkos${VT100_RESET}")

set(BACKEND_KOKKOS_C kokkos.c)

add_library(NablaKokkos ${BACKEND_KOKKOS_C})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hook)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/call)

target_link_libraries(NablaKokkos NablaKokkosHook NablaKokkosCall)

add_dependencies(NablaKokkos libNabla)
