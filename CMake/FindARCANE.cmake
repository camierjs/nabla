include(${NABLA_SOURCE_DIR}/CMake/CMakeTPL.txt)

find_path(ARCANE_INCLUDE_DIR arcane_config.h ${ARCANE_ROOT}/include)
mark_as_advanced(ARCANE_INCLUDE_DIR)

if(ARCANE_INCLUDE_DIR)
  set(ARCANE_FOUND "YES")
endif()