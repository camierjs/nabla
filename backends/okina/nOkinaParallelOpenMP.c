// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
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
