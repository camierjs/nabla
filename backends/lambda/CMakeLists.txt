info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND LAMBDA #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

##################
# Lambda BACKEND #
##################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}Lambda${VT100_RESET}")

set(BACKEND_LAMBDA_C lambda.c
    #hook.c
    #main.c
    #vars.c
    #mesh.c
    #data.c
    #header.c
    tokens.c
    source.c
    #gather.c
    #forall.c
    scatter.c
    #params.c
    reduce.c
    parallelCilk.c
    parallelOpenMP.c
    parallelVoid.c)

add_library(NablaLambda ${BACKEND_LAMBDA_C})
add_dependencies(NablaLambda libNabla)
