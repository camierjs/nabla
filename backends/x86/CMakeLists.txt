info("${VT100_BOLD}${VT100_FG_GREEN}###########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND LIBRARIES #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}###########################${VT100_RESET}")

#####################
# Libraries BACKEND #
#####################
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}LIBRARIES${VT100_RESET} for ${VT100_BOLD}BACKENDS${VT100_FG_YELLOW}${VT100_RESET}")

#################
# Library: CALL #
#################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/call)

#################
# Library: DUMP #
#################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dump)

#################
# Library: HOOK #
#################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hook)

##################
# Library: ALEPH #
##################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/aleph)
if(HYPRE_FOUND)
    set(ALEPH_LIB ${ALEPH_LIB} PARENT_SCOPE)
    set(ALEPH_HYPRE_LIB ${ALEPH_HYPRE_LIB} PARENT_SCOPE)
endif(HYPRE_FOUND)
