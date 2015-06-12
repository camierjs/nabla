info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND LAMBDA #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

##################
# Lambda BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}Lambda${VT100_RESET}")
set(BACKEND_LAMBDA_C 
  nLambdaData.c 
  nLambdaHook.c 
  nLambdaHookGather.c 
  nLambdaHookReductions.c 
  nLambdaMain.c
  nLambdaHookDiffraction.c 
  nLambdaHookScatter.c
  nLambdaMesh.c       
  nLambdaHookArguments.c 
  nLambdaHookForLoops.c 
  nLambdaHookParameters.c 
  nLambdaHookTokens.c 
  nLambdaVariables.c
  ${CMAKE_CURRENT_BINARY_DIR}/nLambdaDump.S)

# On spécifie en plus que nOkinaDump.S va utiliser incbin pour se compiler
set_property(SOURCE nLambdaDump.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/outputs/*.*)

configure_file(nLambdaDump.S ${CMAKE_CURRENT_BINARY_DIR}/nLambdaDump.S)

add_library(NablaLambda ${BACKEND_LAMBDA_C})
add_dependencies(NablaLambda Nabla)