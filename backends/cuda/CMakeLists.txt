info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND CUDA #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}######################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}CUDA${VT100_RESET}")

set(BACKEND_CUDA_C cuda.c
    hook.c
    main.c
    vars.c
    mesh.c
    header.c
    source.c
    forall.c
    reduce.c)

add_library(NablaCuda ${BACKEND_CUDA_C})
add_dependencies(NablaCuda Nabla)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dump)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/cuSolver)

target_link_libraries(NablaCuda NablaCuSolver NablaCudaDump)

#install(TARGETS NablaCuda DESTINATION lib)
