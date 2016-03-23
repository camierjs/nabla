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


// ****************************************************************************
// * Backend OKINA - Génération de la connectivité du maillage coté header
// ****************************************************************************
void xHookMesh1DConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\
\t// ********************************************************\n\
\t// * MESH CONNECTIVITY (1D)\n\
\t// ********************************************************\n\
\tint* xs_cell_node=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*NABLA_NODE_PER_CELL);\n\
\tint* xs_cell_next=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*1);\n\
\tint* xs_cell_prev=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*1);\n\
\tint* xs_cell_face=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*NABLA_FACE_PER_CELL);// is NULL\n\
\tint* xs_node_cell=(int*)calloc(NABLA_NB_NODES,sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* xs_node_cell_corner=(int*)calloc(NABLA_NB_NODES,sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* xs_node_cell_and_corner=(int*)calloc(NABLA_NB_NODES,sizeof(int)*2*NABLA_CELL_PER_NODE);\n\
\tint* xs_face_cell=(int*)calloc(NABLA_NB_FACES,sizeof(int)*NABLA_CELL_PER_FACE);\n\
\tint* xs_face_node=(int*)calloc(NABLA_NB_FACES,sizeof(int)*NABLA_NODE_PER_FACE);\n\
\tassert(xs_cell_node && xs_cell_next && xs_cell_prev && xs_cell_face);\n\
\tassert(xs_node_cell && xs_node_cell_corner && xs_node_cell_and_corner);\n\
\tassert(xs_face_cell && xs_face_node);\n");
}


// ****************************************************************************
// * okinaMesh
// * Adding padding for simd too 
// ****************************************************************************
static void xHookMesh1D(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH GENERATION (1D)\n\
// ********************************************************\n\
const int NABLA_NODE_PER_CELL = 2;\n\
const int NABLA_CELL_PER_NODE = 2;\n\
const int NABLA_CELL_PER_FACE = 0;\n\
const int NABLA_NODE_PER_FACE = 0;\n\
const int NABLA_FACE_PER_CELL = 0;\n\
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
const int NABLA_NB_NODES_WARP = (NABLA_NB_NODES/WARP_SIZE);\n\
const int NABLA_NODES_PADDING = (((NABLA_NB_NODES%%1)==0)?0:1);\n\
const int NABLA_NB_CELLS      = (NABLA_NB_CELLS_X_AXIS);\n\
const int NABLA_NB_CELLS_WARP = (NABLA_NB_CELLS/WARP_SIZE);\n\
\n\
const int NABLA_NB_FACES      = 0;\n\
\n\
int NABLA_NB_PARTICLES /* = NB_PARTICLES*/;\n\
");
}


// ****************************************************************************
// * xHookMesh2DConnectivity
// * Ces connectivités sont maintenant dans le main
// ****************************************************************************
void xHookMesh2DConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\n\
\t// ********************************************************\n\
\t// * MESH CONNECTIVITY (2D)\n\
\t// ********************************************************\n\
\tint* xs_cell_node=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*NABLA_NODE_PER_CELL);\n\
\tint* xs_cell_next=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*2);\n\
\tint* xs_cell_prev=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*2);\n\
\tint* xs_cell_face=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*NABLA_FACE_PER_CELL);\n\
\tint* xs_node_cell=(int*)calloc(NABLA_NB_NODES,sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* xs_node_cell_corner=(int*)calloc(NABLA_NB_NODES,sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* xs_node_cell_and_corner=(int*)calloc(NABLA_NB_NODES,sizeof(int)*2*NABLA_CELL_PER_NODE);\n\
\tint* xs_face_cell=(int*)calloc(NABLA_NB_FACES,sizeof(int)*NABLA_CELL_PER_FACE);\n\
\tint* xs_face_node=(int*)calloc(NABLA_NB_FACES,sizeof(int)*NABLA_NODE_PER_FACE);\n\
\tassert(xs_cell_node && xs_cell_next && xs_cell_prev && xs_cell_face);\n\
\tassert(xs_node_cell && xs_node_cell_corner && xs_node_cell_and_corner);\n\
\tassert(xs_face_cell && xs_face_node);\n");
}

// ****************************************************************************
// * xHookMesh2D
// ****************************************************************************
static void xHookMesh2D(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\
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
const int NABLA_NB_CELLS        = (NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS);\n\
\n\
const int NABLA_NB_NODES_WARP   = (NABLA_NB_NODES/WARP_SIZE);\n\
const int NABLA_NB_CELLS_WARP   = (NABLA_NB_CELLS/WARP_SIZE);\n\
const int NABLA_NB_OUTER_CELLS_WARP   = ((2*(X_EDGE_ELEMS+Y_EDGE_ELEMS)-4)/WARP_SIZE);\n\
int nxtOuterCellOffset(const int c){\n\
  //printf(\"NABLA_NB_OUTER_CELLS_WARP=%%d, NABLA_NB_CELLS_X_AXIS=%%d\",NABLA_NB_OUTER_CELLS_WARP,NABLA_NB_CELLS_X_AXIS);\n\
  if (c<NABLA_NB_CELLS_X_AXIS) {printf(\"1\"); return 1;}\n\
  if (c>=(NABLA_NB_CELLS-NABLA_NB_CELLS_X_AXIS)) {printf(\"4\"); return 1;}\n \
  if ((c%%(NABLA_NB_CELLS_X_AXIS))==0) {printf(\"2\"); return NABLA_NB_CELLS_X_AXIS-1;}\n\
  if (((c+1)%%(NABLA_NB_CELLS_X_AXIS))==0) {printf(\"3\"); return 1;}\n\
  printf(\"else\");\n\
  return NABLA_NB_CELLS_X_AXIS-1;\n\
}\n\
\n\
int NABLA_NB_PARTICLES /*= NB_PARTICLES*/;\n\
");
}

// ****************************************************************************
// * Backend OKINA - Génération de la connectivité du maillage coté header
// ****************************************************************************
void xHookMesh3DConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\
\t// ********************************************************\n\
\t// * MESH CONNECTIVITY (3D)\n\
\t// ********************************************************\n\
\tint* xs_cell_node=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*NABLA_NODE_PER_CELL);\n\
\tint* xs_cell_next=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*3);\n\
\tint* xs_cell_prev=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*3);\n\
\tint* xs_cell_face=(int*)calloc(NABLA_NB_CELLS,sizeof(int)*NABLA_FACE_PER_CELL);\n\
\tint* xs_node_cell=(int*)calloc(NABLA_NB_NODES,sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* xs_node_cell_corner=(int*)calloc(NABLA_NB_NODES,sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* xs_node_cell_and_corner=(int*)calloc(NABLA_NB_NODES,sizeof(int)*2*NABLA_CELL_PER_NODE);\n\
\tint* xs_face_cell=(int*)calloc(NABLA_NB_FACES,sizeof(int)*NABLA_CELL_PER_FACE);\n\
\tint* xs_face_node=(int*)calloc(NABLA_NB_FACES,sizeof(int)*NABLA_NODE_PER_FACE);\n\
\tassert(xs_cell_node && xs_cell_next && xs_cell_prev && xs_cell_face);\n\
\tassert(xs_node_cell && xs_node_cell_corner && xs_node_cell_and_corner);\n\
\tassert(xs_face_cell && xs_face_node);\n");
}


// ****************************************************************************
// ****************************************************************************
void xHookMeshFreeConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\
\t// ********************************************************\n\
\t// * FREE MESH CONNECTIVITY \n\
\t// ********************************************************\n\
\tfree(xs_cell_node);\n\
\tfree(xs_node_cell);\n\
\tfree(xs_node_cell_corner);\n\
\tfree(xs_cell_next);\n\
\tfree(xs_cell_prev);\n\
\tfree(xs_node_cell_and_corner);\n\
\tfree(xs_face_cell);\n\
\tfree(xs_face_node);\n\
\tfree(xs_cell_face);\n");
}


// ****************************************************************************
// * okinaMesh
// * Adding padding for simd too 
// ****************************************************************************
static void xHookMesh3D(nablaMain *nabla){
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
\n\
const int NABLA_NB_NODES_WARP   = (NABLA_NB_NODES/WARP_SIZE);\n\
const int NABLA_NB_CELLS_WARP   = (NABLA_NB_CELLS/WARP_SIZE);\n\
");
}

// ****************************************************************************
// * Backend LAMBDA - Allocation de la connectivité du maillage
// * On a pas encore parsé!, on ne peut pas jouer avec les isWithLibrary
// ****************************************************************************
void xHookMeshPrefix(nablaMain *nabla){
  dbg("\n[lambdaMainMeshPrefix]");
  //fprintf(nabla->entity->src,"\t//[lambdaMainMeshPrefix] Allocation des connectivités");
  dbg("\n[xHookMeshCore] nabla->entity->libraries=0x%X",nabla->entity->libraries);
  // Mesh structures and functions depends on the ℝ library that can be used
}

// ****************************************************************************
// * xHookMeshCore
// * Ici, on revient une fois parsé!
// ****************************************************************************
void xHookMeshCore(nablaMain *nabla){
  dbg("\n[xHookMeshCore]");
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * xHookMeshCore\n\
// ********************************************************\n");
  if (isWithLibrary(nabla,with_real)){
    xHookMesh1D(nabla);
    //xHookMesh1DConnectivity(nabla);
  }else if (isWithLibrary(nabla,with_real2)){
    xHookMesh2D(nabla);
    //xHookMesh2DConnectivity(nabla);
  }else{
    xHookMesh3D(nabla);
    //xHookMesh3DConnectivity(nabla);
  }
  xDumpMesh(nabla);
}


// ****************************************************************************
// * xHookMeshPostfix
// ****************************************************************************
void xHookMeshPostfix(nablaMain *nabla){}
