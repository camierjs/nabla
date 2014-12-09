/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM
 *****************************************************************************
 * File     : nccCudaMesh.c
 * Author   : Camier Jean-Sylvain
 * Created  : 2012.12.13
 * Updated  : 2012.12.13
 *****************************************************************************
 * Description:
 *****************************************************************************
 * Date			Author	Description
 * 2012.12.13	camierjs	Creation
 *****************************************************************************/
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
#warning 1+NABLA_NB_NODES\n\
const int NABLA_NB_NODES_WARP   = (0+NABLA_NB_NODES/WARP_SIZE);\n\
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
