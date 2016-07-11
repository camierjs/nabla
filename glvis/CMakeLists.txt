# Nabla GLVis
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}GLVis${VT100_RESET}")
set(GLVis_C glvis.cpp)

if (MFEM_FOUND)
   include_directories(${MFEM_INCLUDE_PATH})
   add_library(NablaGLVis ${GLVis_C})
endif (MFEM_FOUND)
