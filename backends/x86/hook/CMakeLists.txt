info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}LIBRARY${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}HOOK${VT100_RESET}")

set(BACKEND_LIB_HOOK_C
    hook.c
    header.c
    forall.c
    gather.c
    main.c
    vars.c
    mesh.c
    data.c
    reduce.c
    scatter.c
    source.c
    tokens.c)

add_library(NablaX86Hook ${BACKEND_LIB_HOOK_C})
add_dependencies(NablaX86Hook libNabla)
