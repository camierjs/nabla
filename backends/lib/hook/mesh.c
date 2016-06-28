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
// * 
// ****************************************************************************
void xHookMeshStruct(nablaMain *nabla){
#warning xHookMeshStruct __APPLE__ switches aligned_alloc
  fprintf(nabla->entity->hdr,"\n\n\
#if defined(__APPLE__)\n\
\t#define aligned_alloc(align,size) malloc(size)\n\
#endif\n");
  
  fprintf(nabla->entity->hdr,"\n\n\
// ****************************************************************************\n\
// * nablaMshStruct\n\
// ****************************************************************************\n\
typedef struct nablaMshStruct{\n\
\tint NABLA_NODE_PER_CELL;\n\
\tint NABLA_CELL_PER_NODE;\n\
\tint NABLA_CELL_PER_FACE;\n\
\tint NABLA_NODE_PER_FACE;\n\
\tint NABLA_FACE_PER_CELL;\n\
\n\
\tint NABLA_NB_NODES_X_AXIS;\n\
\tint NABLA_NB_NODES_Y_AXIS;\n\
\tint NABLA_NB_NODES_Z_AXIS;\n\
\n\
\tint NABLA_NB_CELLS_X_AXIS;\n\
\tint NABLA_NB_CELLS_Y_AXIS;\n\
\tint NABLA_NB_CELLS_Z_AXIS;\n\
\n\
\tint NABLA_NB_FACES_X_INNER;\n\
\tint NABLA_NB_FACES_Y_INNER;\n\
\tint NABLA_NB_FACES_Z_INNER;\n\
\tint NABLA_NB_FACES_X_OUTER;\n\
\tint NABLA_NB_FACES_Y_OUTER;\n\
\tint NABLA_NB_FACES_Z_OUTER;\n\
\tint NABLA_NB_FACES_INNER;\n\
\tint NABLA_NB_FACES_OUTER;\n\
\tint NABLA_NB_FACES;\n\
\n\
\tdouble NABLA_NB_NODES_X_TICK;\n\
\tdouble NABLA_NB_NODES_Y_TICK;\n\
\tdouble NABLA_NB_NODES_Z_TICK;\n\
\n\
\tint NABLA_NB_NODES;\n\
\tint NABLA_NODES_PADDING;\n\
\tint NABLA_NB_CELLS;\n\
\tint NABLA_NB_NODES_WARP;\n\
\tint NABLA_NB_CELLS_WARP;\n\
}nablaMesh;\n");
}


// ****************************************************************************
// * Backend OKINA - Génération de la connectivité du maillage coté header
// ****************************************************************************
void xHookMesh1DConnectivity(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\
\t// ********************************************************\n\
\t// * MESH CONNECTIVITY (1D)\n\
\t// ********************************************************\n\
\tint* xs_cell_node=(int*)aligned_alloc(WARP_ALIGN,sizeof(int)*NABLA_NB_CELLS*NABLA_NODE_PER_CELL);\n\
\tint* xs_cell_next=(int*)aligned_alloc(WARP_ALIGN,sizeof(int)*1*NABLA_NB_CELLS);\n\
\tint* xs_cell_prev=(int*)aligned_alloc(WARP_ALIGN,sizeof(int)*1*NABLA_NB_CELLS);\n\
\tint* xs_cell_face=(int*)aligned_alloc(WARP_ALIGN,sizeof(int)*NABLA_FACE_PER_CELL*NABLA_NB_CELLS);// is NULL\n\
\tint* xs_node_cell=(int*)aligned_alloc(WARP_ALIGN,sizeof(int)*NABLA_CELL_PER_NODE*NABLA_NB_NODES);\n\
\tint* xs_node_cell_corner=(int*)aligned_alloc(WARP_ALIGN,sizeof(int)*NABLA_CELL_PER_NODE*NABLA_NB_NODES);\n\
\tint* xs_node_cell_and_corner=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_NODES*sizeof(int)*2*NABLA_CELL_PER_NODE);\n\
\tint* xs_face_cell=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_FACES*sizeof(int)*NABLA_CELL_PER_FACE);\n\
\tint* xs_face_node=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_FACES*sizeof(int)*NABLA_NODE_PER_FACE);\n\
\tassert(xs_cell_node && xs_cell_next && xs_cell_prev && xs_cell_face);\n\
\tassert(xs_node_cell && xs_node_cell_corner && xs_node_cell_and_corner);\n\
\tassert(xs_face_cell && xs_face_node);\n");
}


// ****************************************************************************
// * okinaMesh
// * Adding padding for simd too 
// ****************************************************************************
void xHookMesh1D(nablaMain *nabla){
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
const int NABLA_NB_FACES_X_INNER = 0;\n\
const int NABLA_NB_FACES_Y_INNER = 0;\n\
const int NABLA_NB_FACES_Z_INNER = 0;\n\
const int NABLA_NB_FACES_X_OUTER = 0;\n\
const int NABLA_NB_FACES_Y_OUTER = 0;\n\
const int NABLA_NB_FACES_Z_OUTER = 0;\n\
const int NABLA_NB_FACES_INNER = 0;\n\
const int NABLA_NB_FACES_OUTER = 0;\n\
const int NABLA_NB_FACES = 0;\n\
\n\
const double NABLA_NB_NODES_X_TICK = LENGTH/(NABLA_NB_CELLS_X_AXIS);\n\
const double NABLA_NB_NODES_Y_TICK = 0.0;\n\
const double NABLA_NB_NODES_Z_TICK = 0.0;\n\
\n\
const int NABLA_NB_CELLS      = (NABLA_NB_CELLS_X_AXIS);\n\
const int NABLA_NB_NODES      = (NABLA_NB_NODES_X_AXIS);\n\
const int NABLA_NODES_PADDING = (((NABLA_NB_NODES%%WARP_SIZE)==0)?0:1);\n\
\n\
const int NABLA_NB_NODES_WARP = (NABLA_NODES_PADDING+NABLA_NB_NODES/WARP_SIZE);\n\
const int NABLA_NB_CELLS_WARP = (NABLA_NB_CELLS/WARP_SIZE);\n\
const int NABLA_NB_OUTER_CELLS_WARP = 2;\n\n");
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
\tint* xs_cell_node=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_CELLS*sizeof(int)*NABLA_NODE_PER_CELL);\n\
\tint* xs_cell_next=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_CELLS*sizeof(int)*2);\n\
\tint* xs_cell_prev=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_CELLS*sizeof(int)*2);\n\
\tint* xs_cell_face=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_CELLS*sizeof(int)*NABLA_FACE_PER_CELL);\n\
\tint* xs_node_cell=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_NODES*sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* xs_node_cell_corner=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_NODES*sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* xs_node_cell_and_corner=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_NODES*sizeof(int)*2*NABLA_CELL_PER_NODE);\n\
\tint* xs_face_cell=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_FACES*sizeof(int)*NABLA_CELL_PER_FACE);\n\
\tint* xs_face_node=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_FACES*sizeof(int)*NABLA_NODE_PER_FACE);\n\
\tassert(xs_cell_node && xs_cell_next && xs_cell_prev && xs_cell_face);\n\
\tassert(xs_node_cell && xs_node_cell_corner && xs_node_cell_and_corner);\n\
\tassert(xs_face_cell && xs_face_node);\n");
}

// ****************************************************************************
// * xHookMesh2D
// ****************************************************************************
void xHookMesh2D(nablaMain *nabla){
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
const int NABLA_NB_FACES_Z_INNER = 0;\n\
const int NABLA_NB_FACES_X_OUTER = 2*NABLA_NB_CELLS_Y_AXIS;\n\
const int NABLA_NB_FACES_Y_OUTER = 2*NABLA_NB_CELLS_X_AXIS;\n\
const int NABLA_NB_FACES_Z_OUTER = 0;\n\
const int NABLA_NB_FACES_INNER = NABLA_NB_FACES_X_INNER+NABLA_NB_FACES_Y_INNER;\n\
const int NABLA_NB_FACES_OUTER = NABLA_NB_FACES_X_OUTER+NABLA_NB_FACES_Y_OUTER;\n\
const int NABLA_NB_FACES = NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;\n\
\n\
const double NABLA_NB_NODES_X_TICK = LENGTH/(NABLA_NB_CELLS_X_AXIS);\n\
const double NABLA_NB_NODES_Y_TICK = LENGTH/(NABLA_NB_CELLS_Y_AXIS);\n\
const double NABLA_NB_NODES_Z_TICK = 0.0;\n\
\n\
const int NABLA_NB_NODES        = (NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS);\n\
const int NABLA_NODES_PADDING   = (((NABLA_NB_NODES%%WARP_SIZE)==0)?0:1);\n\
const int NABLA_NB_CELLS        = (NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS);\n\
\n\
const int NABLA_NB_NODES_WARP   = (NABLA_NODES_PADDING+NABLA_NB_NODES/WARP_SIZE);\n\
const int NABLA_NB_CELLS_WARP   = (NABLA_NB_CELLS/WARP_SIZE);\n\
const int NABLA_NB_OUTER_CELLS_WARP   = ((2*(X_EDGE_ELEMS+Y_EDGE_ELEMS)-4)/WARP_SIZE);\n\
int nxtOuterCellOffset(const int c){\n\
  //printf(\"NABLA_NB_OUTER_CELLS_WARP=%%d, NABLA_NB_CELLS_X_AXIS=%%d\",NABLA_NB_OUTER_CELLS_WARP,NABLA_NB_CELLS_X_AXIS);\n\
  if (c<NABLA_NB_CELLS_X_AXIS) {/*printf(\"1\");*/ return 1;}\n\
  if (c>=(NABLA_NB_CELLS-NABLA_NB_CELLS_X_AXIS)) {/*printf(\"4\");*/ return 1;}\n\
  if ((c%%(NABLA_NB_CELLS_X_AXIS))==0) {/*printf(\"2\");*/ return NABLA_NB_CELLS_X_AXIS-1;}\n\
  if (((c+1)%%(NABLA_NB_CELLS_X_AXIS))==0) {/*printf(\"3\");*/ return 1;}\n \
  /*printf(\"else\");*/\n\
  return NABLA_NB_CELLS_X_AXIS-1;\n\
}\n\
\n\
int NABLA_NB_PARTICLES /*= NB_PARTICLES*/;\n\
");
}

// ****************************************************************************
// * Backend OKINA - Génération de la connectivité du maillage coté header
// ****************************************************************************
void xHookMesh3DConnectivity(nablaMain *nabla,const char *pfx){
  fprintf(nabla->entity->src,"\n\
\t// ********************************************************\n\
\t// * MESH CONNECTIVITY (3D) with prefix '%s'\n\
\t// ********************************************************\n\
\tint* %s_cell_node=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_CELLS*sizeof(int)*NABLA_NODE_PER_CELL);\n\
\tint* %s_cell_next=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_CELLS*sizeof(int)*3);\n\
\tint* %s_cell_prev=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_CELLS*sizeof(int)*3);\n\
\tint* %s_cell_face=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_CELLS*sizeof(int)*NABLA_FACE_PER_CELL);\n\
\tint* %s_node_cell=(int*)aligned_alloc(WARP_ALIGN,(NABLA_NB_NODES+NABLA_NODES_PADDING)*sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* %s_node_cell_corner=(int*)aligned_alloc(WARP_ALIGN,(NABLA_NB_NODES+NABLA_NODES_PADDING)*sizeof(int)*NABLA_CELL_PER_NODE);\n\
\tint* %s_node_cell_and_corner=(int*)aligned_alloc(WARP_ALIGN,(NABLA_NB_NODES+NABLA_NODES_PADDING)*sizeof(int)*2*NABLA_CELL_PER_NODE);\n\
\tint* %s_face_cell=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_FACES*sizeof(int)*NABLA_CELL_PER_FACE);\n\
\tint* %s_face_node=(int*)aligned_alloc(WARP_ALIGN,NABLA_NB_FACES*sizeof(int)*NABLA_NODE_PER_FACE);\n\
\tassert(%s_cell_node && %s_cell_next && %s_cell_prev && %s_cell_face);\n\
\tassert(%s_node_cell && %s_node_cell_corner && %s_node_cell_and_corner);\n\
\tassert(%s_face_cell && %s_face_node);\n",
          pfx,pfx,pfx,pfx,pfx,pfx,pfx,pfx,pfx,
          pfx,pfx,pfx,pfx,pfx,pfx,pfx,pfx,pfx,pfx);
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
void xHookMesh3D(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * MESH GENERATION (3D)\n\
// ********************************************************\n\
static const int NABLA_NODE_PER_CELL = 8;\n\
static const int NABLA_CELL_PER_NODE = 8;\n\
static const int NABLA_CELL_PER_FACE = 2;\n\
static const int NABLA_NODE_PER_FACE = 4;\n\
static const int NABLA_FACE_PER_CELL = 6;\n");
  
  fprintf(nabla->entity->src,"\n\n\t\
// ********************************************************\n\t\
// * MESH GENERATION (3D)\n\t\
// ********************************************************\n\t\
const int NABLA_NB_NODES_X_AXIS = X_EDGE_ELEMS+1;\n\t\
const int NABLA_NB_NODES_Y_AXIS = Y_EDGE_ELEMS+1;\n\t\
const int NABLA_NB_NODES_Z_AXIS = Z_EDGE_ELEMS+1;\n\t\
\n\t\
const int NABLA_NB_CELLS_X_AXIS = X_EDGE_ELEMS;\n\t\
const int NABLA_NB_CELLS_Y_AXIS = Y_EDGE_ELEMS;\n\t\
const int NABLA_NB_CELLS_Z_AXIS = Z_EDGE_ELEMS;\n\t\
\n\t\
const int NABLA_NB_FACES_X_INNER = (X_EDGE_ELEMS-1)*Y_EDGE_ELEMS*Z_EDGE_ELEMS;\n\t\
const int NABLA_NB_FACES_Y_INNER = (Y_EDGE_ELEMS-1)*X_EDGE_ELEMS*Z_EDGE_ELEMS;\n\t\
const int NABLA_NB_FACES_Z_INNER = (Z_EDGE_ELEMS-1)*X_EDGE_ELEMS*Y_EDGE_ELEMS;\n\t\
const int NABLA_NB_FACES_X_OUTER = 2*NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS;\n\t\
const int NABLA_NB_FACES_Y_OUTER = 2*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Z_AXIS;\n\t\
const int NABLA_NB_FACES_Z_OUTER = 2*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS;\n\t\
const int NABLA_NB_FACES_INNER = NABLA_NB_FACES_Z_INNER+NABLA_NB_FACES_X_INNER+NABLA_NB_FACES_Y_INNER;\n\t\
const int NABLA_NB_FACES_OUTER = NABLA_NB_FACES_X_OUTER+NABLA_NB_FACES_Y_OUTER+NABLA_NB_FACES_Z_OUTER;\n\t\
const int NABLA_NB_FACES = NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;\n\t\
\n\t\
const double NABLA_NB_NODES_X_TICK = LENGTH/(NABLA_NB_CELLS_X_AXIS);\n\t\
const double NABLA_NB_NODES_Y_TICK = LENGTH/(NABLA_NB_CELLS_Y_AXIS);\n\t\
const double NABLA_NB_NODES_Z_TICK = LENGTH/(NABLA_NB_CELLS_Z_AXIS);\n\t\
\n\t\
const int NABLA_NB_NODES        = (NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS);\n\t\
const int NABLA_NB_CELLS        = (NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS);\n\t\
const int NABLA_NODES_PADDING   = ((NABLA_NB_NODES%%WARP_SIZE)==0?0:WARP_SIZE-NABLA_NB_NODES%%WARP_SIZE);\n\t\
const int NABLA_NODES_PADDING_WARP = (((NABLA_NB_NODES%%WARP_SIZE)==0)?0:1);\n\t\
//const int NABLA_CELLS_PADDING   = (((NABLA_NB_CELLS%%WARP_SIZE)==0)?0:1);\n\t\
const int NABLA_NB_NODES_WARP   = (NABLA_NODES_PADDING_WARP+NABLA_NB_NODES/WARP_SIZE);\n\t\
const int NABLA_NB_CELLS_WARP   = (NABLA_NB_CELLS/WARP_SIZE);\n\t\
//printf(\"NABLA_NODES_PADDING=%%d\\n\",NABLA_NODES_PADDING);\n\
// A verifier:\n\t\
__attribute__((unused)) const int NABLA_NB_OUTER_CELLS_WARP = (((2*X_EDGE_ELEMS*Y_EDGE_ELEMS)+(Z_EDGE_ELEMS-2)*(2*(X_EDGE_ELEMS+Y_EDGE_ELEMS)-4))/WARP_SIZE);\n");
}

// ****************************************************************************
// * Backend LAMBDA - Allocation de la connectivité du maillage
// * On a pas encore parsé!, on ne peut pas jouer avec les isWithLibrary
// ****************************************************************************
void xHookMeshPrefix(nablaMain *nabla){
  dbg("\n[lambdaMainMeshPrefix]");
  //fprintf(nabla->entity->src,"\t//[lambdaMainMeshPrefix] Allocation des connectivités");
  dbg("\n[xHookMeshCore] nabla->entity->libraries=0x%X",nabla->entity->libraries);
}

// ****************************************************************************
// * xHookMeshCore
// * Ici, on revient une fois parsé!
// ****************************************************************************
void xHookMeshCore(nablaMain *nabla){
  dbg("\n[xHookMeshCore]");
  

  
  xDumpMesh(nabla);
}


// ****************************************************************************
// * xHookMeshPostfix
// ****************************************************************************
void xHookMeshPostfix(nablaMain *nabla){}
