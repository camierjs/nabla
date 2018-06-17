info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}NCC${VT100_RESET}")

bison_target(nParser nabla.y ${CMAKE_CURRENT_BINARY_DIR}/nabla.tab.c COMPILE_FLAGS "--report=all")
#--warnings=all,error --graph 
flex_target(nScanner nabla.l ${CMAKE_CURRENT_BINARY_DIR}/nabla.lex.c COMPILE_FLAGS "")
add_flex_bison_dependency(nScanner nParser)

configure_file(nabla.S ${CMAKE_CURRENT_BINARY_DIR}/nabla.asm.S)
add_executable(ncc nabla.c ${CMAKE_CURRENT_BINARY_DIR}/nabla.asm.S)
add_library(Nabla ${BISON_nParser_OUTPUTS} ${FLEX_nScanner_OUTPUTS})

##################
# Link libraries #
##################
target_link_libraries(ncc Nabla NablaMiddlend NablaFrontend NablaTools
								NablaOkina NablaLambda              
								NablaLibCall NablaLibHook NablaLibDump NablaAleph)
if (ARCANE_FOUND)
	target_link_libraries(ncc NablaArcane)
endif (ARCANE_FOUND)

if (CUDA_FOUND)
	target_link_libraries(ncc NablaCuda)
endif(CUDA_FOUND)

if (RAJA_FOUND)
	target_link_libraries(ncc NablaRaja)
endif (RAJA_FOUND)

if (KOKKOS_FOUND)
	target_link_libraries(ncc NablaKokkos)
endif (KOKKOS_FOUND)

if (LEGION_FOUND)
	target_link_libraries(ncc NablaLegion)
endif (LEGION_FOUND)

###########
# INSTALL #
###########
install(TARGETS ncc DESTINATION bin)
