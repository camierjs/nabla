info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_MAGENTA}LIBRARY${VT100_RESET}: CUDA/${VT100_BOLD}${VT100_FG_MAGENTA}CuSparse${VT100_RESET}")

set(CUSPARSE_SOURCES
  nCuSparse.c
  ${CMAKE_CURRENT_BINARY_DIR}/nCuSparse.S)

configure_file(nCuSparse.S ${CMAKE_CURRENT_BINARY_DIR}/nCuSparse.S)
                
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/nCuSparse.S 
  PROPERTY OBJECT_DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_library(NablaCuSparse ${CUSPARSE_SOURCES})
add_dependencies(NablaCuSparse NablaCudaHook)
