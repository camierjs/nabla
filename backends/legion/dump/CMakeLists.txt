set(BACKEND_LEGION_DUMP ${CMAKE_CURRENT_BINARY_DIR}/legion.S)

# On spécifie en plus que legion.S va utiliser incbin pour se compiler
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/legion.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/legion/*.*
             ${CMAKE_CURRENT_SOURCE_DIR}/legion/makefile
             ${CMAKE_CURRENT_SOURCE_DIR}/pennant/*.*)
configure_file(legion.S ${CMAKE_CURRENT_BINARY_DIR}/legion.S)

add_library(NablaLegionDump ${BACKEND_LEGION_DUMP})
