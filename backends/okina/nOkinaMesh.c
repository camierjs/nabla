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


//****************************************************************************
// * Backend OKINA - Génération de la connectivité du maillage coté header
// ***************************************************************************
static void okinaMeshConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\n\
// ********************************************************\n\
// * MESH CONNECTIVITY\n\
// ********************************************************\
\nint cell_node[8*NABLA_NB_CELLS]         __attribute__ ((aligned(WARP_ALIGN)));\
\nint node_cell[8*NABLA_NB_NODES]         __attribute__ ((aligned(WARP_ALIGN)));\
\nint node_cell_corner[8*NABLA_NB_NODES]  __attribute__ ((aligned(WARP_ALIGN)));\
\nint cell_next[3*NABLA_NB_CELLS]         __attribute__ ((aligned(WARP_ALIGN)));\
\nint cell_prev[3*NABLA_NB_CELLS]         __attribute__ ((aligned(WARP_ALIGN)));\
\nint node_cell_and_corner[2*8*NABLA_NB_NODES]         __attribute__ ((aligned(WARP_ALIGN)));\
\n\n\n");
}



// ****************************************************************************
// * okinaMesh
// * Adding padding for simd too 
// ****************************************************************************
void okinaMesh(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH GENERATION\n\
// ********************************************************\n\
const int NABLA_NB_NODES_X_AXIS = X_EDGE_ELEMS+1;\n\
const int NABLA_NB_NODES_Y_AXIS = Y_EDGE_ELEMS+1;\n\
const int NABLA_NB_NODES_Z_AXIS = Z_EDGE_ELEMS+1;\n\
\n\
const int NABLA_NB_CELLS_X_AXIS = X_EDGE_ELEMS;\n\
const int NABLA_NB_CELLS_Y_AXIS = Y_EDGE_ELEMS;\n\
const int NABLA_NB_CELLS_Z_AXIS = Z_EDGE_ELEMS;\n\
\n\
const double NABLA_NB_NODES_X_TICK = LENGTH/(NABLA_NB_CELLS_X_AXIS);\n\
const double NABLA_NB_NODES_Y_TICK = LENGTH/(NABLA_NB_CELLS_Y_AXIS);\n\
const double NABLA_NB_NODES_Z_TICK = LENGTH/(NABLA_NB_CELLS_Z_AXIS);\n\
\n\
const int NABLA_NB_NODES        = (NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS);\n\
const int NABLA_NODES_PADDING   = (((NABLA_NB_NODES%%WARP_SIZE)==0)?0:1);\n\
const int NABLA_NB_NODES_WARP   = (NABLA_NODES_PADDING+NABLA_NB_NODES/WARP_SIZE);\n\
const int NABLA_NB_CELLS        = (NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS);\n \
const int NABLA_NB_CELLS_WARP   = (NABLA_NB_CELLS/WARP_SIZE);");
  okinaMeshConnectivity(nabla);
}


/*****************************************************************************
 * Backend OKINA - Allocation de la connectivité du maillage
 *****************************************************************************/
void nccOkinaMainMeshPrefix(nablaMain *nabla){
  dbg("\n[nccOkinaMainMeshPrefix]");
  fprintf(nabla->entity->src,"\t// [nccOkinaMainMeshPrefix] Allocation des connectivités");
}


void nccOkinaMainMeshPostfix(nablaMain *nabla){
  dbg("\n[nccOkinaMainMeshPostfix]");
  //fprintf(nabla->entity->src,"");
}
