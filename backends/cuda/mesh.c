///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
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


static void cuHookMesh3D(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH GENERATION (3D)\n\
// ********************************************************\n\
#define NABLA_NODE_PER_CELL 8\n\
#define NABLA_CELL_PER_NODE 8\n\
#define NABLA_CELL_PER_FACE 2\n\
#define NABLA_NODE_PER_FACE 4\n\
\n\
#define NABLA_NB_NODES_X_AXIS   (X_EDGE_ELEMS+1)\n\
#define NABLA_NB_NODES_Y_AXIS   (Y_EDGE_ELEMS+1)\n\
#define NABLA_NB_NODES_Z_AXIS   (Z_EDGE_ELEMS+1)\n\
\n\
#define NABLA_NB_CELLS_X_AXIS    X_EDGE_ELEMS\n\
#define NABLA_NB_CELLS_Y_AXIS    Y_EDGE_ELEMS\n\
#define NABLA_NB_CELLS_Z_AXIS    Z_EDGE_ELEMS\n\
\n\
#define NABLA_NB_FACES_X_INNER ((X_EDGE_ELEMS-1)*Y_EDGE_ELEMS*Z_EDGE_ELEMS)\n\
#define NABLA_NB_FACES_Y_INNER ((Y_EDGE_ELEMS-1)*X_EDGE_ELEMS*Z_EDGE_ELEMS)\n\
#define NABLA_NB_FACES_Z_INNER ((Z_EDGE_ELEMS-1)*X_EDGE_ELEMS*Y_EDGE_ELEMS)\n\
#define NABLA_NB_FACES_X_OUTER (2*NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS)\n\
#define NABLA_NB_FACES_Y_OUTER (2*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Z_AXIS)\n\
#define NABLA_NB_FACES_Z_OUTER (2*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS)\n\
#define NABLA_NB_FACES_INNER (NABLA_NB_FACES_Z_INNER+NABLA_NB_FACES_X_INNER+NABLA_NB_FACES_Y_INNER)\n\
#define NABLA_NB_FACES_OUTER (NABLA_NB_FACES_X_OUTER+NABLA_NB_FACES_Y_OUTER+NABLA_NB_FACES_Z_OUTER)\n\
#define NABLA_NB_FACES (NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER)\n\
\n\
#define BLOCKSIZE                  128\n\
#define CUDA_NB_THREADS_PER_BLOCK  128\n\
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

// ****************************************************************************
// * Backend CUDA - Génération de la connectivité du maillage coté header
// ****************************************************************************
static void nLambdaHookMesh3DDeviceVariables(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH CONNECTIVITY\n\
// ********************************************************\n\
__builtin_align__(8) int *xs_cell_node;\n\
__builtin_align__(8) int *xs_node_cell;\n\
__builtin_align__(8) int *xs_node_cell_corner;\n\
__builtin_align__(8) int *xs_node_cell_corner_idx;\n\
__builtin_align__(8) int *xs_cell_next;\n\
__builtin_align__(8) int *xs_cell_prev;\n\
__builtin_align__(8) int *xs_node_cell_and_corner;\n\
__builtin_align__(8) int *xs_face_cell;\n\
__builtin_align__(8) int *xs_face_node;\n\
\n");
}


// ****************************************************************************
// * cuHookMeshPrefix
// ****************************************************************************
void cuHookMeshPrefix(nablaMain *nabla){
  dbg("\n[cuHookMeshPrefix]");
  dbg("\n[cuHookMeshPrefix] nabla->entity->libraries=0x%X",nabla->entity->libraries);
  // Mesh structures and functions depends on the ℝ library that can be used
  if (isWithLibrary(nabla,with_real)){
    //cuHookMesh1D(nabla);
  }else{
    cuHookMesh3D(nabla);
    nLambdaHookMesh3DDeviceVariables(nabla);
  }
}





/*****************************************************************************
 * Backend CUDA - Génération de la connectivité du maillage coté main
 *****************************************************************************/
static void nLambdaHookMesh3DConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\n\
/******************************************************************************\n\
 * Kernel d'initialisation du maillage à-la-SOD\n\
 ******************************************************************************/\n\
__global__ void nabla_ini_node_coords(int *node_cell,\n\
                                      int *node_cell_corner,\n\
                                      int *node_cell_corner_idx,\n\
                                      %s){\n\
\tCUDA_INI_NODE_THREAD(n);\n\
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
__global__ void nabla_ini_cell_connectivity(int *xs_cell_node){\n\
  CUDA_INI_CELL_THREAD(c);\n\
  const int iX=c%%NABLA_NB_CELLS_X_AXIS;\n\
  const int iY=((c/NABLA_NB_CELLS_X_AXIS)%%NABLA_NB_CELLS_Y_AXIS);\n\
  const int iZ=((c/(NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS))%%NABLA_NB_CELLS_Z_AXIS);\n\
  //const int cell_uid=iX+iY*NABLA_NB_CELLS_X_AXIS+iZ*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS;\n \
  const int node_bid=iX+iY*NABLA_NB_NODES_X_AXIS+iZ*NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;\n\
  xs_cell_node[c+0*NABLA_NB_CELLS]=node_bid;\n\
  xs_cell_node[c+1*NABLA_NB_CELLS]=node_bid +1;\n\
  xs_cell_node[c+2*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS + 1;\n\
  xs_cell_node[c+3*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS + 0;\n\
  xs_cell_node[c+4*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;\n\
  xs_cell_node[c+5*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS+1;\n\
  xs_cell_node[c+6*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS+NABLA_NB_NODES_X_AXIS+1;\n\
  xs_cell_node[c+7*NABLA_NB_CELLS]=node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS+NABLA_NB_NODES_X_AXIS+0;\n}\n",
          "real3 *node_coord",
          "\tnode_coord[n]=real3(dx,dy,dz);");
}



// ****************************************************************************
// * cuHookMeshCore
// ****************************************************************************
void cuHookMeshCore(nablaMain *nabla){
  nLambdaHookMesh3DConnectivity(nabla);
}


// ****************************************************************************
// * Backend CUDA - Allocation de la connectivité du maillage
// ****************************************************************************
void cuHookMeshConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\n\
\t// Allocation coté CUDA des connectivités aux mailles\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_cell_node, 8*NABLA_NB_CELLS*sizeof(int)));\n\
\t// Allocation coté CUDA des connectivités aux noeuds\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_node_cell, 8*NABLA_NB_NODES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_node_cell_corner, 8*NABLA_NB_NODES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_node_cell_corner_idx, NABLA_NB_NODES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_cell_next, 3*NABLA_NB_CELLS*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_cell_prev, 3*NABLA_NB_CELLS*sizeof(int)));\n\
\t//CUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_node_cell_and_corner, 2*8*NABLA_NB_NODES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_face_cell, NABLA_CELL_PER_FACE*NABLA_NB_FACES*sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_face_node, NABLA_NODE_PER_FACE*NABLA_NB_FACES*sizeof(int)));\n\
\n\
");
}
