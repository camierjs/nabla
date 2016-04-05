info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# NABLA BACKEND ARCANE #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}########################${VT100_RESET}")

include_directories(${NABLA_BINARY_DIR})

# BACKEND ARCANE
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}ARCANE${VT100_RESET}")
set(BACKEND_ARCANE_C arcane.c
  axl.c hlt.c job.c main.c source.c mesh.c vars.c
  hooks.c module.c family.c
  jCells.c jNodes.c jFaces.c jParticles.c
  tokens.c data.c reduce.c header.c params.c)
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
  lib/gmp.c
  lib/dft.c
  lib/mma.c
  lib/mail.c
  lib/aleph.c
  lib/slurm.c
  lib/scheme.c
  lib/cartesian.c 
  lib/materials.c
  lib/particles.c)

add_library(NablaArcane ${BACKEND_ARCANE_C} ${LIBRARIES_ARCANE_C})

add_dependencies(NablaArcane libNabla)
