include(${NABLA_SOURCE_DIR}/CMake/CMakeTPL.txt)

find_path(HYPRE_INCLUDE_DIR HYPRE.h ${HYPRE_ROOT}/include)
find_library(HYPRE_LIB HYPRE ${HYPRE_ROOT}/lib)

set(HYPRE_FOUND "NO" )
if(HYPRE_INCLUDE_DIR)
  if(HYPRE_LIB)
    set(HYPRE_FOUND "YES" )
  endif()
endif()
