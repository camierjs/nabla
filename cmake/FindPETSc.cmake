find_path(PETSC_INCLUDE_DIRS petsc.h)

find_library(PETSC_LIB petsc)
find_library(PETSC_KSP_LIB petscksp)
find_library(PETSC_MAT_LIB petscmat)
find_library(PETSC_VEC_LIB petscvec)
find_library(PETSC_DM_LIB petscdm)

if (PETSC_LIB)
   set(PETSC_LIBRARIES ${PETSC_LIB})
endif()

if (PETSC_KSP_LIB)
   set(PETSC_LIBRARIES ${PETSC_LIBRARIES} ${PETSC_KSP_LIB})
endif()

if (PETSC_MAT_LIB)
   set(PETSC_LIBRARIES ${PETSC_LIBRARIES} ${PETSC_MAT_LIB})
endif()

if (PETSC_VEC_LIB)
   set(PETSC_LIBRARIES ${PETSC_LIBRARIES} ${PETSC_VEC_LIB})
endif()

if (PETSC_DM_LIB)
   set(PETSC_LIBRARIES ${PETSC_LIBRARIES} ${PETSC_DM_LIB})
endif()

set(PETSC_FOUND "NO" )
if(PETSC_INCLUDE_DIRS)
  if(PETSC_LIBRARIES)
    set(PETSC_FOUND "YES" )
  endif()
endif()
