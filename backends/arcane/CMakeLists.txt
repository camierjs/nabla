info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND ARCANE #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

include_directories(${NABLA_BINARY_DIR})

# BACKEND ARCANE
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}ARCANE${VT100_RESET}")
set(BACKEND_ARCANE_C
  nArcane.c
  nArcaneAXL.c
  nArcaneJob.c
  nArcaneMain.c
  nArcaneSimd.c
  nArcaneHooks.c
  nArcaneConfig.c
  nArcaneModule.c
  nArcaneFunction.c 
  nArcaneJobCells.c
  nArcaneJobNodes.c
  nArcaneJobFaces.c
  nArcaneJobParticles.c)
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
set(LIBRARIES_ARCANE_C
  libraries/nArcaneLibGMP.c
  libraries/nArcaneLibDFT.c
  libraries/nArcaneLibMail.c
  libraries/nArcaneLibAleph.c
  libraries/nArcaneLibSlurm.c
  libraries/nArcaneLibScheme.c
  libraries/nArcaneLibCartesian.c 
  libraries/nArcaneLibMaterials.c
  libraries/nArcaneLibParticles.c
  libraries/nArcaneLibMathematica.c)

add_library(NablaArcane ${BACKEND_ARCANE_C} ${LIBRARIES_ARCANE_C})
add_dependencies(NablaArcane libNabla)
