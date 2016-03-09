set(BACKEND_OKINA_DUMP_C ${CMAKE_CURRENT_BINARY_DIR}/okina.S)

# On spécifie en plus que nOkinaDump.S va utiliser incbin pour se compiler
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/okina.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/dump/avx/*.*
             ${CMAKE_CURRENT_SOURCE_DIR}/dump/mic/*.*
             ${CMAKE_CURRENT_SOURCE_DIR}/dump/sse/*.*
             ${CMAKE_CURRENT_SOURCE_DIR}/dump/std/*.*)
configure_file(okina.S ${CMAKE_CURRENT_BINARY_DIR}/okina.S)

add_library(NablaOkinaCall ${BACKEND_OKINA_DUMP_C})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/simd)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/para)
target_link_libraries(NablaOkinaCall NablaOkinaPara NablaOkinaSimd)

add_dependencies(NablaOkinaCall NablaOkina)
