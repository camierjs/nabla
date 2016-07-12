include(${NABLA_SOURCE_DIR}/cmake/CMakeTPL.txt)

find_path(HYPRE_INCLUDE_DIR HYPRE.h ${HYPRE_ROOT}/include)

# We are looking for libHYPRE.a (sequential!)
find_library(HYPRE_LIB libHYPRE.a ${HYPRE_ROOT}/lib NO_DEFAULT_PATH)

set(HYPRE_FOUND "NO" )

if(HYPRE_INCLUDE_DIR AND HYPRE_LIB)
  set(HYPRE_FOUND "YES" )
endif(HYPRE_INCLUDE_DIR AND HYPRE_LIB)

if(HYPRE_FOUND)
  info("${VT100_FG_MAGENTA}HYPRE${VT100_RESET} has been found in\
        ${VT100_BOLD}${HYPRE_ROOT}${VT100_RESET}")
  
endif(HYPRE_FOUND)
