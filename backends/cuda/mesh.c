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


static void cuHookMesh3D(nablaMain *nabla){}

// ****************************************************************************
// * Backend CUDA - Génération de la connectivité du maillage coté header
// ****************************************************************************
void nLambdaHookMesh3DDeviceVariables(nablaMain *nabla){}


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
    //nLambdaHookMesh3DDeviceVariables(nabla);
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
  
  fprintf(nabla->entity->src,"\t// cuHookMeshConnectivity");
  
  if (isWithLibrary(nabla,with_real))
    xHookMesh1D(nabla);
  else if (isWithLibrary(nabla,with_real2))
    xHookMesh2D(nabla);
  else
    xHookMesh3D(nabla);
  
  fprintf(nabla->entity->src,"\
\tconst nablaMesh msh={\n\
\t\tNABLA_NODE_PER_CELL,\n\
\t\tNABLA_CELL_PER_NODE,\n\
\t\tNABLA_CELL_PER_FACE,\n\
\t\tNABLA_NODE_PER_FACE,\n\
\t\tNABLA_FACE_PER_CELL,\n\
\n\
\t\tNABLA_NB_NODES_X_AXIS,\n\
\t\tNABLA_NB_NODES_Y_AXIS,\n\
\t\tNABLA_NB_NODES_Z_AXIS,\n\
\n\
\t\tNABLA_NB_CELLS_X_AXIS,\n\
\t\tNABLA_NB_CELLS_Y_AXIS,\n\
\t\tNABLA_NB_CELLS_Z_AXIS,\n\
\n\
\t\tNABLA_NB_FACES_X_INNER,\n\
\t\tNABLA_NB_FACES_Y_INNER,\n\
\t\tNABLA_NB_FACES_Z_INNER,\n\
\t\tNABLA_NB_FACES_X_OUTER,\n\
\t\tNABLA_NB_FACES_Y_OUTER,\n\
\t\tNABLA_NB_FACES_Z_OUTER,\n\
\t\tNABLA_NB_FACES_INNER,\n\
\t\tNABLA_NB_FACES_OUTER,\n\
\t\tNABLA_NB_FACES,\n\
\n\
\t\tNABLA_NB_NODES_X_TICK,\n\
\t\tNABLA_NB_NODES_Y_TICK,\n\
\t\tNABLA_NB_NODES_Z_TICK,\n\
\n\
\t\tNABLA_NB_NODES,\n\
\t\tNABLA_NODES_PADDING,\n\
\t\tNABLA_NB_CELLS,\n\
\t\tNABLA_NB_NODES_WARP,\n\
\t\tNABLA_NB_CELLS_WARP};\n");
  
  xHookMesh3DConnectivity(nabla,"host");

  fprintf(nabla->entity->src,"\n\t__builtin_align__(8)real3* host_node_coord=(real3*)calloc(NABLA_NB_NODES,sizeof(real3));// WARP_ALIGN\n\tnabla_ini_connectivity(msh,host_node_coord,\n\t\t\t\t\t\t\t\t\thost_cell_node,host_cell_prev,host_cell_next,host_cell_face,\n\t\t\t\t\t\t\t\t\thost_node_cell,host_node_cell_corner,host_node_cell_and_corner,\n\t\t\t\t\t\t\t\t\thost_face_cell,host_face_node);");
  
  fprintf(nabla->entity->src,"\n\n\
\t__builtin_align__(8) int* xs_cell_node;\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_cell_node, NABLA_NB_CELLS*NABLA_NODE_PER_CELL*sizeof(int)));\n\
\t__builtin_align__(8) int* xs_cell_next;\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_cell_next, NABLA_NB_CELLS*3*sizeof(int)));\n\
\t__builtin_align__(8) int* xs_cell_prev;\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_cell_prev, NABLA_NB_CELLS*3*sizeof(int)));\n\
\t__builtin_align__(8) int* xs_cell_face;\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_cell_face, NABLA_NB_CELLS*NABLA_FACE_PER_CELL*sizeof(int)));\n\
\t__builtin_align__(8) int* xs_node_cell;\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_node_cell, NABLA_NB_NODES*NABLA_CELL_PER_NODE*sizeof(int)));\n\
\t__builtin_align__(8) int* xs_node_cell_corner;\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_node_cell_corner, NABLA_NB_NODES*NABLA_CELL_PER_NODE*sizeof(int)));\n\
\t//__builtin_align__(8) int* xs_node_cell_corner_idx;\n\
\t//CUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_node_cell_corner_idx, NABLA_NB_NODES*sizeof(int)));\n\
\t__builtin_align__(8) int* xs_node_cell_and_corner;\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_node_cell_and_corner, NABLA_NB_NODES*2*NABLA_CELL_PER_NODE*sizeof(int)));\n\
\t__builtin_align__(8) int* xs_face_cell;\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_face_cell, NABLA_CELL_PER_FACE*NABLA_NB_FACES*sizeof(int)));\n\
\t__builtin_align__(8) int* xs_face_node;\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&xs_face_node, NABLA_NODE_PER_FACE*NABLA_NB_FACES*sizeof(int)));\n\
\n\
");
}
