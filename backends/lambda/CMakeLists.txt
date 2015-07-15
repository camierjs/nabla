info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND LAMBDA #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

##################
# Lambda BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}Lambda${VT100_RESET}")
set(BACKEND_LAMBDA_C 
  nLambda.c 
  nLambdaData.c 
  nLambdaMain.c
  nLambdaVariables.c)
# ${CMAKE_CURRENT_BINARY_DIR}/nLambdaDump.S)

# On spécifie en plus que nLambdaDump.S va utiliser incbin pour se compiler
set_property(SOURCE nLambdaDump.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/dump/*.*)

configure_file(nLambdaDump.S ${CMAKE_CURRENT_BINARY_DIR}/nLambdaDump.S)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hooks)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mesh)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/parallel)

add_library(NablaLambda ${BACKEND_LAMBDA_C})
add_dependencies(NablaLambda Nabla)