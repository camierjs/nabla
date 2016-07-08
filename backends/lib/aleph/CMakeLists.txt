info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_MAGENTA}LIBRARY${VT100_RESET}: ${VT100_BOLD}${VT100_FG_MAGENTA}NablaAleph${VT100_RESET}")

set(NABLA_ALEPH_SOURCES nablaAleph.c ${CMAKE_CURRENT_BINARY_DIR}/nablaAleph.S)
                
configure_file(nablaAleph.S ${CMAKE_CURRENT_BINARY_DIR}/nablaAleph.S)
                
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/nablaAleph.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/*.*)


#configure_file(Aleph.h ${CMAKE_CURRENT_BINARY_DIR}/Aleph.h)
#configure_file(AlephTypesSolver.h ${CMAKE_CURRENT_BINARY_DIR}/AlephTypesSolver.h)

add_library(NablaLibAleph ${NABLA_ALEPH_SOURCES})
