info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: LAMBDA/${VT100_BOLD}${VT100_FG_YELLOW}DUMP${VT100_RESET}")

set(BACKEND_LAMBDA_DUMP_C
  nLambdaDump.c
  ${CMAKE_CURRENT_BINARY_DIR}/nLambdaDump.S)

# On spécifie en plus que nLambdaDump.S va utiliser incbin pour se compiler
set_property(SOURCE
  ${CMAKE_CURRENT_BINARY_DIR}/nLambdaDump.S 
  PROPERTY OBJECT_DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

configure_file(nLambdaDump.S ${CMAKE_CURRENT_BINARY_DIR}/nLambdaDump.S)

add_library(NablaLambdaDump ${BACKEND_LAMBDA_DUMP_C})

# We add a dependency in order to make sure the "nabla.tab.h" exists
add_dependencies(NablaLambdaDump NablaLambdaHook)

