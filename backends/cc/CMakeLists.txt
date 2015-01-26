info("${VT100_BOLD}${VT100_FG_GREEN}#####################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND *CC #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}#####################${VT100_RESET}")

# BACKEND *cc
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}*CC${VT100_RESET}")
set(BACKEND_CC_C 
    ccData.c
    ccHook.c
    ccJobs.c
    ccMain.c
    ccMesh.c
    ${CMAKE_CURRENT_BINARY_DIR}/ccDump.S)

# On spécifie en plus que nOkinaDump.S va utiliser incbin pour se compiler
set_property(SOURCE ccDump.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/outputs/*.*)

configure_file(ccDump.S ${CMAKE_CURRENT_BINARY_DIR}/ccDump.S)

add_library(NablaCc ${BACKEND_CC_C})
add_dependencies(NablaCc Nabla)