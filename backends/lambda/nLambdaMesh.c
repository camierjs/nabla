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


//****************************************************************************
// * Backend CC - G�n�ration de la connectivit� du maillage cot� header
// ***************************************************************************
static void ccMeshConnectivity(nablaMain *nabla){
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
// * ccMesh
// * Adding padding for simd too 
// ****************************************************************************
void ccMesh(nablaMain *nabla){
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
  ccMeshConnectivity(nabla);
}


/*****************************************************************************
 * Backend CC - Allocation de la connectivit� du maillage
 *****************************************************************************/
void nccCcMainMeshPrefix(nablaMain *nabla){
  dbg("\n[nccCcMainMeshPrefix]");
  fprintf(nabla->entity->src,"\t// [nccCcMainMeshPrefix] Allocation des connectivit�s");
}


void nccCcMainMeshPostfix(nablaMain *nabla){
  dbg("\n[nccCcMainMeshPostfix]");
  //fprintf(nabla->entity->src,"");
}