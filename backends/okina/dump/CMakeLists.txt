set(BACKEND_OKINA_DUMP_C ${CMAKE_CURRENT_BINARY_DIR}/okina.S)

# On spécifie en plus que nOkinaDump.S va utiliser incbin pour se compiler
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/okina.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/avx/*.*
             ${CMAKE_CURRENT_SOURCE_DIR}/mic/*.*
             ${CMAKE_CURRENT_SOURCE_DIR}/sse/*.*
             ${CMAKE_CURRENT_SOURCE_DIR}/std/*.*)
configure_file(okina.S ${CMAKE_CURRENT_BINARY_DIR}/okina.S)

add_library(NablaOkinaDump ${BACKEND_OKINA_DUMP_C})
