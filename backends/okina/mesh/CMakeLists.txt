info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: OKINA/${VT100_BOLD}${VT100_FG_YELLOW}MESH${VT100_RESET}")
set(BACKEND_OKINA_MESH_C nOkinaMesh1D.c nOkinaMesh3D.c nOkinaMesh.c)

add_library(NablaOkinaMesh ${BACKEND_OKINA_MESH_C})

#add_dependencies(NablaOkinaSimd Nabla)
