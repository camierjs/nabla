info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}OPTORG${VT100_RESET}")
bison_target(oParser orgopt.y ${CMAKE_CURRENT_BINARY_DIR}/orgopt.tab.c COMPILE_FLAGS "--report=all")
flex_target(oScanner orgopt.l ${CMAKE_CURRENT_BINARY_DIR}/orgopt.lex.c)
add_flex_bison_dependency(oScanner oParser)

add_library(OrgOpt ${BISON_oParser_OUTPUTS}
            ${FLEX_oScanner_OUTPUTS}
            ${NABLA_SOURCE_DIR}/toolbox/utf8.c
            ${NABLA_SOURCE_DIR}/frontend/astNode.c
            ${NABLA_SOURCE_DIR}/frontend/dbg.c)

add_executable(orgopt orgopt.c)

target_link_libraries(orgopt OrgOpt NablaFrontend NablaToolbox)
