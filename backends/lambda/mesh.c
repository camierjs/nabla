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
#include "backends/lambda/lambda.h"
#include "backends/x86/dump/dump.h"


// ****************************************************************************
// * Backend OKINA - Génération de la connectivité du maillage coté header
// ****************************************************************************
static void nLambdaHookMesh1DConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\n\
// ********************************************************\n\
// * MESH CONNECTIVITY (1D)\n\
// ********************************************************\
\nint cell_node[NABLA_NODE_PER_CELL*NABLA_NB_CELLS];\
\nint node_cell[NABLA_NODE_PER_CELL*NABLA_NB_NODES];\
\nint node_cell_corner[NABLA_NODE_PER_CELL*NABLA_NB_NODES];\
\nint cell_next[1*NABLA_NB_CELLS];\
\nint cell_prev[1*NABLA_NB_CELLS];\
\nint node_cell_and_corner[2*NABLA_CELL_PER_NODE*NABLA_NB_NODES];\
\nint *face_cell=NULL;\
\nint *face_node=NULL;\
\n\n\n");
}


// ****************************************************************************
// * okinaMesh
// * Adding padding for simd too 
// ****************************************************************************
static void nLambdaHookMesh1D(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH GENERATION (1D)\n\
// ********************************************************\n\
const int NABLA_NODE_PER_CELL = 2;\n\
const int NABLA_CELL_PER_NODE = 2;\n\
const int NABLA_CELL_PER_FACE = 0;\n\
const int NABLA_NODE_PER_FACE = 0;\n\
\n\
const int NABLA_NB_NODES_X_AXIS = X_EDGE_ELEMS+1;\n\
const int NABLA_NB_NODES_Y_AXIS = 1;\n\
const int NABLA_NB_NODES_Z_AXIS = 1;\n\
\n\
const int NABLA_NB_CELLS_X_AXIS = X_EDGE_ELEMS;\n\
const int NABLA_NB_CELLS_Y_AXIS = 1;\n\
const int NABLA_NB_CELLS_Z_AXIS = 1;\n\
\n\
const double NABLA_NB_NODES_X_TICK = LENGTH/(NABLA_NB_CELLS_X_AXIS);\n\
const double NABLA_NB_NODES_Y_TICK = 0.0;\n\
const double NABLA_NB_NODES_Z_TICK = 0.0;\n\
\n\
const int NABLA_NB_NODES      = (NABLA_NB_NODES_X_AXIS);\n\
const int NABLA_NODES_PADDING = (((NABLA_NB_NODES%%1)==0)?0:1);\n\
const int NABLA_NB_CELLS      = (NABLA_NB_CELLS_X_AXIS);\n\
\n\
const int NABLA_NB_FACES         = 0;\n\
\n\
int NABLA_NB_PARTICLES /* = NB_PARTICLES*/;\n\
");
}


// ****************************************************************************
// * nLambdaHookMesh2DConnectivity
// ****************************************************************************
static void nLambdaHookMesh2DConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\n\
// ********************************************************\n\
// * MESH CONNECTIVITY (2D)\n\
// ********************************************************\
\nint cell_node[NABLA_NODE_PER_CELL*NABLA_NB_CELLS];\
\nint cell_next[2*NABLA_NB_CELLS];\
\nint cell_prev[2*NABLA_NB_CELLS];\
\nint cell_face[NABLA_FACE_PER_CELL*NABLA_NB_CELLS];\
\nint node_cell[NABLA_CELL_PER_NODE*NABLA_NB_NODES];\
\nint node_cell_corner[NABLA_CELL_PER_NODE*NABLA_NB_NODES];\
\nint node_cell_and_corner[2*NABLA_CELL_PER_NODE*NABLA_NB_NODES];\
\nint face_cell[NABLA_CELL_PER_FACE*NABLA_NB_FACES];\
\nint face_node[NABLA_NODE_PER_FACE*NABLA_NB_FACES];\
\n\n\n");
}


// ****************************************************************************
// * nLambdaHookMesh2D
// ****************************************************************************
static void nLambdaHookMesh2D(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH GENERATION (2D)\n\
// ********************************************************\n\
const int NABLA_NODE_PER_CELL = 4;\n\
const int NABLA_CELL_PER_NODE = 4;\n\
const int NABLA_CELL_PER_FACE = 2;\n\
const int NABLA_NODE_PER_FACE = 2;\n\
const int NABLA_FACE_PER_CELL = 4;\n\
\n\
const int NABLA_NB_NODES_X_AXIS = X_EDGE_ELEMS+1;\n\
const int NABLA_NB_NODES_Y_AXIS = Y_EDGE_ELEMS+1;\n\
const int NABLA_NB_NODES_Z_AXIS = 1;\n\
\n\
const int NABLA_NB_CELLS_X_AXIS = X_EDGE_ELEMS;\n\
const int NABLA_NB_CELLS_Y_AXIS = Y_EDGE_ELEMS;\n\
const int NABLA_NB_CELLS_Z_AXIS = 1;\n\
\n\
const int NABLA_NB_FACES_X_INNER = (X_EDGE_ELEMS-1)*Y_EDGE_ELEMS;\n\
const int NABLA_NB_FACES_Y_INNER = (Y_EDGE_ELEMS-1)*X_EDGE_ELEMS;\n\
const int NABLA_NB_FACES_X_OUTER = 2*NABLA_NB_CELLS_Y_AXIS;\n\
const int NABLA_NB_FACES_Y_OUTER = 2*NABLA_NB_CELLS_X_AXIS;\n\
const int NABLA_NB_FACES_INNER = NABLA_NB_FACES_X_INNER+NABLA_NB_FACES_Y_INNER;\n\
const int NABLA_NB_FACES_OUTER = NABLA_NB_FACES_X_OUTER+NABLA_NB_FACES_Y_OUTER;\n\
const int NABLA_NB_FACES = NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;\n\
\n\
const double NABLA_NB_NODES_X_TICK = LENGTH/(NABLA_NB_CELLS_X_AXIS);\n\
const double NABLA_NB_NODES_Y_TICK = LENGTH/(NABLA_NB_CELLS_Y_AXIS);\n\
\n\
const int NABLA_NB_NODES        = (NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS);\n\
const int NABLA_NODES_PADDING   = (((NABLA_NB_NODES%%1)==0)?0:1);\n\
const int NABLA_NB_CELLS        = (NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS);\n \
\n\
int NABLA_NB_PARTICLES /*= NB_PARTICLES*/;\n\
");
}


// ****************************************************************************
// * Backend OKINA - Génération de la connectivité du maillage coté header
// ****************************************************************************
static void nLambdaHookMesh3DConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\n\
// ********************************************************\n\
// * MESH CONNECTIVITY (3D)\n\
// ********************************************************\
\nint cell_node[NABLA_NODE_PER_CELL*NABLA_NB_CELLS];\
\nint node_cell[NABLA_CELL_PER_NODE*NABLA_NB_NODES];\
\nint node_cell_corner[NABLA_CELL_PER_NODE*NABLA_NB_NODES];\
\nint cell_next[3*NABLA_NB_CELLS];\
\nint cell_prev[3*NABLA_NB_CELLS];\
\nint node_cell_and_corner[2*NABLA_CELL_PER_NODE*NABLA_NB_NODES];\
\nint face_cell[NABLA_CELL_PER_FACE*NABLA_NB_FACES];\
\nint face_node[NABLA_NODE_PER_FACE*NABLA_NB_FACES];\
\nint cell_face[NABLA_FACE_PER_CELL*NABLA_NB_CELLS];\
\n\n\n");
}


// ****************************************************************************
// * okinaMesh
// * Adding padding for simd too 
// ****************************************************************************
static void nLambdaHookMesh3D(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH GENERATION (3D)\n\
// ********************************************************\n\
const int NABLA_NODE_PER_CELL = 8;\n\
const int NABLA_CELL_PER_NODE = 8;\n\
const int NABLA_CELL_PER_FACE = 2;\n\
const int NABLA_NODE_PER_FACE = 4;\n\
const int NABLA_FACE_PER_CELL = 6;\n\
\n\
const int NABLA_NB_NODES_X_AXIS = X_EDGE_ELEMS+1;\n\
const int NABLA_NB_NODES_Y_AXIS = Y_EDGE_ELEMS+1;\n\
const int NABLA_NB_NODES_Z_AXIS = Z_EDGE_ELEMS+1;\n\
\n\
const int NABLA_NB_CELLS_X_AXIS = X_EDGE_ELEMS;\n\
const int NABLA_NB_CELLS_Y_AXIS = Y_EDGE_ELEMS;\n\
const int NABLA_NB_CELLS_Z_AXIS = Z_EDGE_ELEMS;\n\
\n\
const int NABLA_NB_FACES_X_INNER = (X_EDGE_ELEMS-1)*Y_EDGE_ELEMS*Z_EDGE_ELEMS;\n\
const int NABLA_NB_FACES_Y_INNER = (Y_EDGE_ELEMS-1)*X_EDGE_ELEMS*Z_EDGE_ELEMS;\n\
const int NABLA_NB_FACES_Z_INNER = (Z_EDGE_ELEMS-1)*X_EDGE_ELEMS*Y_EDGE_ELEMS;\n\
const int NABLA_NB_FACES_X_OUTER = 2*NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS;\n\
const int NABLA_NB_FACES_Y_OUTER = 2*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Z_AXIS;\n\
const int NABLA_NB_FACES_Z_OUTER = 2*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS;\n\
const int NABLA_NB_FACES_INNER = NABLA_NB_FACES_Z_INNER+NABLA_NB_FACES_X_INNER+NABLA_NB_FACES_Y_INNER;\n\
const int NABLA_NB_FACES_OUTER = NABLA_NB_FACES_X_OUTER+NABLA_NB_FACES_Y_OUTER+NABLA_NB_FACES_Z_OUTER;\n\
const int NABLA_NB_FACES = NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;\n\
\n\
const double NABLA_NB_NODES_X_TICK = LENGTH/(NABLA_NB_CELLS_X_AXIS);\n\
const double NABLA_NB_NODES_Y_TICK = LENGTH/(NABLA_NB_CELLS_Y_AXIS);\n\
const double NABLA_NB_NODES_Z_TICK = LENGTH/(NABLA_NB_CELLS_Z_AXIS);\n\
\n\
const int NABLA_NB_NODES        = (NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS);\n\
const int NABLA_NODES_PADDING   = (((NABLA_NB_NODES%%1)==0)?0:1);\n\
const int NABLA_NB_CELLS        = (NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS);\n \
\n\
int NABLA_NB_PARTICLES /*= NB_PARTICLES*/;\n\
");
}


// ****************************************************************************
// * Backend LAMBDA - Allocation de la connectivité du maillage
// * On a pas encore parsé!, on ne peut pas jouer avec les isWithLibrary
// ****************************************************************************
void nLambdaHookMeshPrefix(nablaMain *nabla){
  dbg("\n[lambdaMainMeshPrefix]");
  //fprintf(nabla->entity->src,"\t//[lambdaMainMeshPrefix] Allocation des connectivités");
  dbg("\n[nLambdaHookMeshCore] nabla->entity->libraries=0x%X",nabla->entity->libraries);
  // Mesh structures and functions depends on the ℝ library that can be used
}


// ****************************************************************************
// * nLambdaHookMeshCore
// * Ici, on revient une fois parsé!
// ****************************************************************************
void nLambdaHookMeshCore(nablaMain *nabla){
  dbg("\n[nLambdaHookMeshCore]");
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * nLambdaHookMeshCore\n\
// ********************************************************\n");
  if (isWithLibrary(nabla,with_real)){
    nLambdaHookMesh1D(nabla);
    nLambdaHookMesh1DConnectivity(nabla);
  }else if (isWithLibrary(nabla,with_real2)){
    nLambdaHookMesh2D(nabla);
    nLambdaHookMesh2DConnectivity(nabla);
  }else{
    nLambdaHookMesh3D(nabla);
    nLambdaHookMesh3DConnectivity(nabla);
  }
  dumpMesh(nabla);
}


// ****************************************************************************
// * nLambdaHookMeshPostfix
// ****************************************************************************
void nLambdaHookMeshPostfix(nablaMain *nabla){
  dbg("\n[nLambdaHookMeshPostfix]");
}
