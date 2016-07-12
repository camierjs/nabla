info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND LAMBDA #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

##################
# Lambda BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}Lambda${VT100_RESET}")

set(BACKEND_LAMBDA_C lambda.c)

add_library(NablaLambda ${BACKEND_LAMBDA_C})
add_dependencies(NablaLambda Nabla)
#install(TARGETS NablaLambda DESTINATION lib)
