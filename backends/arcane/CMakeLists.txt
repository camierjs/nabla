#info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND ARCANE #${VT100_RESET}")
#info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

include_directories(${NABLA_BINARY_DIR})

# BACKEND ARCANE
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}ARCANE${VT100_RESET}")
set(BACKEND_ARCANE_C nccArc.c nccArcConfig.c nccArcModule.c
							nccArcAxl.c nccArcFunction.c 
                     nccArcJob.c nccArcJobCells.c
                     nccArcJobNodes.c nccArcJobFaces.c nccArcJobParticles.c
                     nccArcMain.c nccArcHooks.c nccArcSimd.c)
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}ARCANE${VT100_RESET} libraries:${VT100_RESET}")
info("\t${VT100_FG_YELLOW}GMP${VT100_RESET}")
info("\t${VT100_FG_YELLOW}DFT${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Mail${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Slurm${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Aleph${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Scheme${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Cartesian${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Materials${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Particles${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Mathematica${VT100_RESET}")
set(LIBRARIES_ARCANE_C 	libraries/nccArcLibAleph.c
								libraries/nccArcLibScheme.c
								libraries/nccArcLibCartesian.c 
								libraries/nccArcLibMaterials.c
								libraries/nccArcLibGmp.c
								libraries/nccArcLibDft.c
								libraries/nccArcLibParticles.c
								libraries/nccArcLibMathematica.c 
								libraries/nccArcLibSlurm.c
								libraries/nccArcLibMail.c)

add_library(NablaArcane ${BACKEND_ARCANE_C} ${LIBRARIES_ARCANE_C})
add_dependencies(NablaArcane Nabla)
