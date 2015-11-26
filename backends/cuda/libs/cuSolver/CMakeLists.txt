info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_MAGENTA}LIBRARY${VT100_RESET}: CUDA/${VT100_BOLD}${VT100_FG_MAGENTA}CuSolver${VT100_RESET}")

set(CUSOLVER_SOURCES
  cuSolver.c
  ${CMAKE_CURRENT_BINARY_DIR}/cuSolver.S)

configure_file(cuSolver.S ${CMAKE_CURRENT_BINARY_DIR}/cuSolver.S)
                
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/cuSolver.S 
  PROPERTY OBJECT_DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.*)

add_library(NablaCuSolver ${CUSOLVER_SOURCES})
add_dependencies(NablaCuSolver NablaCudaHook)
