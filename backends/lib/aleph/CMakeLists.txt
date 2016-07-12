info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_MAGENTA}LIBRARY${VT100_RESET}: ${VT100_BOLD}${VT100_FG_MAGENTA}NablaAleph${VT100_RESET}")

set(NABLA_ALEPH_SOURCES nablaAleph.c ${CMAKE_CURRENT_BINARY_DIR}/nablaAleph.S)
                
configure_file(nablaAleph.S ${CMAKE_CURRENT_BINARY_DIR}/nablaAleph.S)
                
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/nablaAleph.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_library(NablaAleph ${NABLA_ALEPH_SOURCES})
#install(TARGETS NablaAleph DESTINATION lib)
