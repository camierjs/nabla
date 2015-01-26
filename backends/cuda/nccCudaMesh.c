///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
// IDDN.FR.001.520002.000.S.P.2014.000.10500                                 //
//                                                                           //
// Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          //
//                                                                           //
// This software is a computer program whose purpose is to translate         //
// numerical-analysis specific sources and to generate optimized code        //
// for different targets and architectures.                                  //
//                                                                           //
// This software is governed by the CeCILL license under French law and      //
// abiding by the rules of distribution of free software. You can  use,      //
// modify and/or redistribute the software under the terms of the CeCILL     //
// license as circulated by CEA, CNRS and INRIA at the following URL:        //
// "http://www.cecill.info".                                                 //
//                                                                           //
// The CeCILL is a free software license, explicitly compatible with         //
// the GNU GPL.                                                              //
//                                                                           //
// As a counterpart to the access to the source code and rights to copy,     //
// modify and redistribute granted by the license, users are provided only   //
// with a limited warranty and the software's author, the holder of the      //
// economic rights, and the successive licensors have only limited liability.//
//                                                                           //
// In this respect, the user's attention is drawn to the risks associated    //
// with loading, using, modifying and/or developing or reproducing the       //
// software by the user in light of its specific status of free software,    //
// that may mean that it is complicated to manipulate, and that also         //
// therefore means that it is reserved for developers and experienced        //
// professionals having in-depth computer knowledge. Users are therefore     //
// encouraged to load and test the software's suitability as regards their   //
// requirements in conditions enabling the security of their systems and/or  //
// data to be ensured and, more generally, to use and operate it in the      //
// same conditions as regards security.                                      //
//                                                                           //
// The fact that you are presently reading this means that you have had      //
// knowledge of the CeCILL license and that you accept its terms.            //
//                                                                           //
// See the LICENSE file for details.                                         //
///////////////////////////////////////////////////////////////////////////////
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
__builtin_align__(8) int *node_cell_corner_idx;\n\
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
                                      int *node_cell_corner_idx,\n\
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
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&cell_node, 8*NABLA_NB_CELLS*sizeof(int)));\n\
\t// Allocation coté CUDA des connectivités aux noeuds\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&node_cell, 8*NABLA_NB_NODES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&node_cell_corner, 8*NABLA_NB_NODES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&node_cell_corner_idx, NABLA_NB_NODES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&cell_next, 3*NABLA_NB_CELLS*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&cell_prev, 3*NABLA_NB_CELLS*sizeof(int)));\n\
\t//CUDA_HANDLE_ERROR(cudaCalloc((void**)&node_cell_and_corner, 2*8*NABLA_NB_NODES*sizeof(int)));\n\
\n\
");
}


void nccCudaMainMeshPostfix(nablaMain *nabla){
  dbg("\n[nccCudaMainMeshPostfix]");
  fprintf(nabla->entity->src,"\n\n\
\tCUDA_HANDLE_ERROR(cudaFree(cell_node));\n\
\tCUDA_HANDLE_ERROR(cudaFree(node_cell));\n\
\tCUDA_HANDLE_ERROR(cudaFree(node_cell_corner));\n\
\tCUDA_HANDLE_ERROR(cudaFree(node_cell_corner_idx));\n\
\tCUDA_HANDLE_ERROR(cudaFree(cell_next));\n\
\tCUDA_HANDLE_ERROR(cudaFree(cell_prev));\n\
\tCUDA_HANDLE_ERROR(cudaFree(node_cell_and_corner));\n\
");
}
