info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: KOKKOS/${VT100_BOLD}${VT100_FG_YELLOW}DUMP${VT100_RESET}")

set(BACKEND_KOKKOS_DUMP_C
  kokkos.c
  ${CMAKE_CURRENT_BINARY_DIR}/kokkos.S)

# On spécifie en plus que nKokkosDump.S va utiliser incbin pour se compiler
set_property(SOURCE
  ${CMAKE_CURRENT_BINARY_DIR}/kokkos.S 
  PROPERTY OBJECT_DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

configure_file(kokkos.S ${CMAKE_CURRENT_BINARY_DIR}/kokkos.S)

add_library(NablaKokkosDump ${BACKEND_KOKKOS_DUMP_C})

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaKokkosDump NablaKokkosHook)

