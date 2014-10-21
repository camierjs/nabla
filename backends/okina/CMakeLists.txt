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
    nOkinaParallelCilk.c
    nOkinaParallelOpenMP.c
    nOkinaParallelVoid.c
    nOkinaDump.S)
# On spécifie en plus que nOkinaDump.S va utiliser incbin pour se compiler
set_property(SOURCE nOkinaDump.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/outputs/*.*)

configure_file(outputs/knDbg.h ${CMAKE_CURRENT_BINARY_DIR}/knDbg.h)
configure_file(outputs/knMsh.c ${CMAKE_CURRENT_BINARY_DIR}/knMsh.c)
configure_file(outputs/knMth.h ${CMAKE_CURRENT_BINARY_DIR}/knMth.h)

configure_file(outputs/knStdReal.h ${CMAKE_CURRENT_BINARY_DIR}/knStdReal.h)
configure_file(outputs/knStdReal3.h ${CMAKE_CURRENT_BINARY_DIR}/knStdReal3.h)
configure_file(outputs/knStdInteger.h ${CMAKE_CURRENT_BINARY_DIR}/knStdInteger.h)
configure_file(outputs/knStdTernary.h ${CMAKE_CURRENT_BINARY_DIR}/knStdTernary.h)
configure_file(outputs/knStdGather.h ${CMAKE_CURRENT_BINARY_DIR}/knStdGather.h)
configure_file(outputs/knStdScatter.h ${CMAKE_CURRENT_BINARY_DIR}/knStdScatter.h)
configure_file(outputs/knStdOStream.h ${CMAKE_CURRENT_BINARY_DIR}/knStdOStream.h)

configure_file(outputs/knSseReal.h ${CMAKE_CURRENT_BINARY_DIR}/knSseReal.h)
configure_file(outputs/knSseReal3.h ${CMAKE_CURRENT_BINARY_DIR}/knSseReal3.h)
configure_file(outputs/knSseInteger.h ${CMAKE_CURRENT_BINARY_DIR}/knSseInteger.h)
configure_file(outputs/knSseTernary.h ${CMAKE_CURRENT_BINARY_DIR}/knSseTernary.h)
configure_file(outputs/knSseGather.h ${CMAKE_CURRENT_BINARY_DIR}/knSseGather.h)
configure_file(outputs/knSseScatter.h ${CMAKE_CURRENT_BINARY_DIR}/knSseScatter.h)
configure_file(outputs/knSseOStream.h ${CMAKE_CURRENT_BINARY_DIR}/knSseOStream.h)

configure_file(outputs/knAvxReal.h ${CMAKE_CURRENT_BINARY_DIR}/knAvxReal.h)
configure_file(outputs/knAvxReal3.h ${CMAKE_CURRENT_BINARY_DIR}/knAvxReal3.h)
configure_file(outputs/knAvxInteger.h ${CMAKE_CURRENT_BINARY_DIR}/knAvxInteger.h)
configure_file(outputs/knAvxTernary.h ${CMAKE_CURRENT_BINARY_DIR}/knAvxTernary.h)
configure_file(outputs/knAvxGather.h ${CMAKE_CURRENT_BINARY_DIR}/knAvxGather.h)
configure_file(outputs/knAvxScatter.h ${CMAKE_CURRENT_BINARY_DIR}/knAvxScatter.h)
configure_file(outputs/knAvxOStream.h ${CMAKE_CURRENT_BINARY_DIR}/knAvxOStream.h)

configure_file(outputs/knMicReal.h ${CMAKE_CURRENT_BINARY_DIR}/knMicReal.h)
configure_file(outputs/knMicReal3.h ${CMAKE_CURRENT_BINARY_DIR}/knMicReal3.h)
configure_file(outputs/knMicInteger.h ${CMAKE_CURRENT_BINARY_DIR}/knMicInteger.h)
configure_file(outputs/knMicTernary.h ${CMAKE_CURRENT_BINARY_DIR}/knMicTernary.h)
configure_file(outputs/knMicGather.h ${CMAKE_CURRENT_BINARY_DIR}/knMicGather.h)
configure_file(outputs/knMicScatter.h ${CMAKE_CURRENT_BINARY_DIR}/knMicScatter.h)
configure_file(outputs/knMicOStream.h ${CMAKE_CURRENT_BINARY_DIR}/knMicOStream.h)

add_library(NablaOkina ${BACKEND_OKINA_C})


info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}OKINA${VT100_RESET} libraries:${VT100_RESET}")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libraries/arcane)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libraries/aleph)
