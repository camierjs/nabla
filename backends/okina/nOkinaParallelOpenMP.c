#include "nabla.h"

// ****************************************************************************
// * OpenMP Sync
// ****************************************************************************
char *nccOkinaParallelOpenMPSync(void){
  return "";//#pragma omp barrier\n";
}


// ****************************************************************************
// * OpenMP Spawn
// ****************************************************************************
char *nccOkinaParallelOpenMPSpawn(void){
  return "";//#pragma omp spawn ";
}


// ****************************************************************************
// * OpenMP for loop
// ****************************************************************************
char *nccOkinaParallelOpenMPLoop(struct nablaMainStruct *n){
  return "\\\n_Pragma(\"omp parallel for firstprivate(NABLA_NB_CELLS,NABLA_NB_CELLS_WARP,NABLA_NB_NODES)\")\\\n";
  //return "\\\n_Pragma(\"ivdep\")\\\n_Pragma(\"vector aligned\")\\\n_Pragma(\"omp parallel for firstprivate(NABLA_NB_CELLS,NABLA_NB_CELLS_WARP,NABLA_NB_NODES)\")\\\n";
}


// ****************************************************************************
// * OpenMP includes
// ****************************************************************************
char *nccOkinaParallelOpenMPIncludes(void){
  return "#include <omp.h>\n";
}
