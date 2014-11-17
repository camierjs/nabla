#info("${VT100_BOLD}${VT100_FG_GREEN}#######################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND OKINA #${VT100_RESET}")
#info("${VT100_BOLD}${VT100_FG_GREEN}#######################${VT100_RESET}")


# BACKEND OKINA
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}OKINA${VT100_RESET}")
set(BACKEND_OKINA_C 
    nOkina.c
    nOkinaSimdStd.c
    nOkinaSimdSse.c
    nOkinaSimdAvx.c
    nOkinaSimdMic.c
    nOkinaData.c
    nOkinaEnum.c
    nOkinaFunc.c
    nOkinaJobs.c
    nOkinaMain.c
    nOkinaMesh.c
    nOkinaPragmas.c
    nOkinaParallelCilk.c
    nOkinaParallelOpenMP.c
    nOkinaParallelVoid.c
    ${CMAKE_CURRENT_BINARY_DIR}/nOkinaDump.S)

# On spécifie en plus que nOkinaDump.S va utiliser incbin pour se compiler
set_property(SOURCE nOkinaDump.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/outputs/*.*)

configure_file(nOkinaDump.S ${CMAKE_CURRENT_BINARY_DIR}/nOkinaDump.S)

add_library(NablaOkina ${BACKEND_OKINA_C})
add_dependencies(NablaOkina Nabla)

info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}OKINA${VT100_RESET} libraries:${VT100_RESET}")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libraries/aleph)
