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


void cudaMesh(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH GENERATION\n\
// ********************************************************\n\
#define NABLA_NB_NODES_X_AXIS   (X_EDGE_ELEMS+1)\n\
#define NABLA_NB_NODES_Y_AXIS   (Y_EDGE_ELEMS+1)\n\
#define NABLA_NB_NODES_Z_AXIS   (Z_EDGE_ELEMS+1)\n\
\n\
#define NABLA_NB_CELLS_X_AXIS    X_EDGE_ELEMS\n\
#define NABLA_NB_CELLS_Y_AXIS    Y_EDGE_ELEMS\n\
#define NABLA_NB_CELLS_Z_AXIS    Z_EDGE_ELEMS\n\
\n\
#define BLOCKSIZE                  256\n\
#define CUDA_NB_THREADS_PER_BLOCK  256\n\
\n\
#define NABLA_NB_NODES_X_TICK LENGTH/(NABLA_NB_CELLS_X_AXIS)\n\
#define NABLA_NB_NODES_Y_TICK LENGTH/(NABLA_NB_CELLS_Y_AXIS)\n\
#define NABLA_NB_NODES_Z_TICK LENGTH/(NABLA_NB_CELLS_Z_AXIS)\n\
\n\
#define NABLA_NB_NODES (NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS)\n\
#define NABLA_NB_CELLS (NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS)\n\
\n\
#define NABLA_NB_GLOBAL 1\n");
}


/*****************************************************************************
 * Backend CUDA - Génération de la connectivité du maillage coté header
 *****************************************************************************/
void cudaMeshConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH CONNECTIVITY\n\
// ********************************************************\n\
__builtin_align__(8) int *cell_node;\n\
__builtin_align__(8) int *node_cell;\n\
__builtin_align__(8) int *node_cell_corner;\n\
__builtin_align__(8) int *cell_next;\n\
__builtin_align__(8) int *cell_prev;\n\
__builtin_align__(8) int *node_cell_and_corner;\n\
\n");
}


/*****************************************************************************
 * Backend CUDA - Génération de la connectivité du maillage coté main
 *****************************************************************************/
void nccCudaMainMeshConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\n\
/******************************************************************************\n\
 * Kernel d'initialisation du maillage à-la-SOD\n\
 ******************************************************************************/\n\
__global__ void nabla_ini_node_coords(int *node_cell,\n\
                                      int *node_cell_corner,\n\
                                      %s){\n\
\tCUDA_INI_NODE_THREAD(tnid);\n\
\tconst int n=tnid;\n\
\tnode_cell[n+0*NABLA_NB_NODES]=-1;\n\
\tnode_cell[n+1*NABLA_NB_NODES]=-1;\n\
\tnode_cell[n+2*NABLA_NB_NODES]=-1;\n\
\tnode_cell[n+3*NABLA_NB_NODES]=-1;\n\
\tnode_cell[n+4*NABLA_NB_NODES]=-1;\n\
\tnode_cell[n+5*NABLA_NB_NODES]=-1;\n\
\tnode_cell[n+6*NABLA_NB_NODES]=-1;\n\
\tnode_cell[n+7*NABLA_NB_NODES]=-1;\n\
\tnode_cell_corner[n+0*NABLA_NB_NODES]=-1;\n\
\tnode_cell_corner[n+1*NABLA_NB_NODES]=-1;\n\
\tnode_cell_corner[n+2*NABLA_NB_NODES]=-1;\n\
\tnode_cell_corner[n+3*NABLA_NB_NODES]=-1;\n\
\tnode_cell_corner[n+4*NABLA_NB_NODES]=-1;\n\
\tnode_cell_corner[n+5*NABLA_NB_NODES]=-1;\n\
\tnode_cell_corner[n+6*NABLA_NB_NODES]=-1;\n\
\tnode_cell_corner[n+7*NABLA_NB_NODES]=-1;\n\
\tconst double dx=((double)(n%%NABLA_NB_NODES_X_AXIS))*NABLA_NB_NODES_X_TICK;\n\
\tconst double dy=((double)((n/NABLA_NB_NODES_X_AXIS)%%NABLA_NB_NODES_Y_AXIS))*NABLA_NB_NODES_Y_TICK;\n\
\tconst double dz=((double)((n/(NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS))%%NABLA_NB_NODES_Z_AXIS))*NABLA_NB_NODES_Z_TICK;\n%s\n\
\t//printf(\"\\n\tNode #%%d @ (%%f,%%f,%%f)\",n,dx,dy,dz);\n\
}\n\
\n\
__global__ void nabla_ini_cell_connectivity(int *cell_node){\n\
  CUDA_INI_CELL_THREAD(tcid);\n\
  const int c=tcid;\n\
  const int iX=c%%NABLA_NB_CELLS_X_AXIS;\n\
  const int iY=((c/NABLA_NB_CELLS_X_AXIS)%%NABLA_NB_CELLS_Y_AXIS);\n\
  const int iZ=((c/(NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS))%%NABLA_NB_CELLS_Z_AXIS);\n\
  //const int cell_uid=iX+iY*NABLA_NB_CELLS_X_AXIS+iZ*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS;\n \
  const int node_bid=iX+iY*NABLA_NB_NODES_X_AXIS+iZ*NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;\n\
  cell_node[tcid+0*NABLA_NB_CELLS]=node_bid;\n\
  cell_node[tcid+1*NABLA_NB_CELLS]=node_bid +1;\n\
  cell_node[tcid+2*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS + 1;\n\
  cell_node[tcid+3*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS + 0;\n\
  cell_node[tcid+4*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;\n\
  cell_node[tcid+5*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS+1;\n\
  cell_node[tcid+6*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS+NABLA_NB_NODES_X_AXIS+1;\n\
  cell_node[tcid+7*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS+NABLA_NB_NODES_X_AXIS+0;\n\
}\n",((nabla->colors&BACKEND_COLOR_OKINA_SOA)==BACKEND_COLOR_OKINA_SOA)?
          "Real *node_coordx, Real *node_coordy, Real *node_coordz":"Real3 *node_coord",
          ((nabla->colors&BACKEND_COLOR_OKINA_SOA)==BACKEND_COLOR_OKINA_SOA)?"\
\tnode_coordx[tnid]=dx;\n\
\tnode_coordy[tnid]=dy;\n\
\tnode_coordz[tnid]=dz;\n":"\tnode_coord[tnid]=Real3(dx,dy,dz);");
}



/*****************************************************************************
 * Backend CUDA - Allocation de la connectivité du maillage
 *****************************************************************************/
void nccCudaMainMeshPrefix(nablaMain *nabla){
  dbg("\n[nccCudaMainMeshPrefix]");
  fprintf(nabla->entity->src,"\n\n\
\t// Allocation coté CUDA des connectivités aux mailles\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&cell_node, 8*NABLA_NB_CELLS*sizeof(int)));\n\
\t// Allocation coté CUDA des connectivités aux noeuds\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&node_cell, 8*NABLA_NB_NODES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&node_cell_corner, 8*NABLA_NB_NODES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&cell_next, 3*NABLA_NB_CELLS*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&cell_prev, 3*NABLA_NB_CELLS*sizeof(int)));\n\
\t//CUDA_HANDLE_ERROR(cudaMalloc((void**)&node_cell_and_corner, 2*8*NABLA_NB_NODES*sizeof(int)));\n\
\n\
");
}


void nccCudaMainMeshPostfix(nablaMain *nabla){
  dbg("\n[nccCudaMainMeshPostfix]");
  fprintf(nabla->entity->src,"\n\n\
\tCUDA_HANDLE_ERROR(cudaFree(cell_node));\n\
\tCUDA_HANDLE_ERROR(cudaFree(node_cell));\n\
\tCUDA_HANDLE_ERROR(cudaFree(node_cell_corner));\n\
\tCUDA_HANDLE_ERROR(cudaFree(cell_next));\n\
\tCUDA_HANDLE_ERROR(cudaFree(cell_prev));\n\
\tCUDA_HANDLE_ERROR(cudaFree(node_cell_and_corner));\n\
");
}
