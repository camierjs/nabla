info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: OKINA/${VT100_BOLD}${VT100_FG_YELLOW}SIMD${VT100_RESET}")
set(BACKEND_OKINA_SIMD_C nOkinaSimdAvx.c  nOkinaSimdMic.c  nOkinaSimdSse.c  nOkinaSimdStd.c)

add_library(NablaOkinaCallSimd ${BACKEND_OKINA_SIMD_C})

add_dependencies(NablaOkinaCallSimd NablaOkina)
