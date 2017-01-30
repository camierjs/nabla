info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND LEGION #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

##################
# Legion BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}Legion${VT100_RESET}")

set(BACKEND_LEGION_C legion.c call.c forall.c gram.c header.c main.c source.c token.c vars.c)

add_library(NablaLegion ${BACKEND_LEGION_C})
add_dependencies(NablaLegion Nabla)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dump)

target_link_libraries(NablaLegion NablaLegionDump)
