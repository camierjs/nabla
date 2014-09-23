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
#define NABLA_NB_NODES_X_TICK (LENGTH/(NABLA_NB_NODES_X_AXIS-1))\n\
#define NABLA_NB_NODES_Y_TICK (LENGTH/(NABLA_NB_NODES_Y_AXIS-1))\n\
#define NABLA_NB_NODES_Z_TICK (LENGTH/(NABLA_NB_NODES_Z_AXIS-1))\n\
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
__builtin_align__(8) int *node_cell;\n\
__builtin_align__(8) int *cell_node;\n");
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
                                      %s){\n\
\tCUDA_INI_NODE_THREAD(tnid);\n\
\tnode_cell[tnid+0*NABLA_NB_NODES]=0;\n\
\tnode_cell[tnid+1*NABLA_NB_NODES]=0;\n\
\tnode_cell[tnid+2*NABLA_NB_NODES]=0;\n\
\tnode_cell[tnid+3*NABLA_NB_NODES]=0;\n\
\tnode_cell[tnid+4*NABLA_NB_NODES]=0;\n\
\tnode_cell[tnid+5*NABLA_NB_NODES]=0;\n\
\tnode_cell[tnid+6*NABLA_NB_NODES]=0;\n\
\tnode_cell[tnid+7*NABLA_NB_NODES]=0;\n\
\tint dx,dy,dz,iNode=tnid;\n\
\tdx=iNode/(NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS);\n\
\tdz=(iNode/NABLA_NB_NODES_Y_AXIS)%%(NABLA_NB_NODES_Z_AXIS);\n\
\tdy=iNode%%NABLA_NB_NODES_Y_AXIS;\n\
%s\n\
}\n\
\n\
__global__ void nabla_ini_cell_connectivity(int *cell_node){\n\
  CUDA_INI_CELL_THREAD(tcid);\n\
  int dx,dy,dz,bid,iCell=tcid;\n\
  dx=iCell/(NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS);\n\
  dz=(iCell/NABLA_NB_CELLS_Y_AXIS)%%(NABLA_NB_CELLS_Z_AXIS);\n\
  dy=iCell%%NABLA_NB_CELLS_Y_AXIS;\n\
  bid=dy+dz*NABLA_NB_NODES_Y_AXIS+dx*NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS;\n\
  cell_node[tcid+0*NABLA_NB_CELLS]=bid;\n\
  cell_node[tcid+1*NABLA_NB_CELLS]=bid+1;\n\
  cell_node[tcid+2*NABLA_NB_CELLS]=bid+NABLA_NB_NODES_Y_AXIS+1;\n\
  cell_node[tcid+3*NABLA_NB_CELLS]=bid+NABLA_NB_NODES_Y_AXIS+0;\n\
  cell_node[tcid+4*NABLA_NB_CELLS]=bid+NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS;\n\
  cell_node[tcid+5*NABLA_NB_CELLS]=bid+NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS+1;\n\
  cell_node[tcid+6*NABLA_NB_CELLS]=bid+NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS+NABLA_NB_NODES_Y_AXIS+1;\n\
  cell_node[tcid+7*NABLA_NB_CELLS]=bid+NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS+NABLA_NB_NODES_Y_AXIS+0;\n\
}\n",((nabla->colors&BACKEND_COLOR_OKINA_SOA)==BACKEND_COLOR_OKINA_SOA)?
          "Real *node_coordx, Real *node_coordy, Real *node_coordz":
          "Real3 *node_coord",
          ((nabla->colors&BACKEND_COLOR_OKINA_SOA)==BACKEND_COLOR_OKINA_SOA)?
          "\tnode_coordx[tnid]=((double)dx)*NABLA_NB_NODES_X_TICK;\n\
\tnode_coordy[tnid]=((double)dy)*NABLA_NB_NODES_Y_TICK;\n\
\tnode_coordz[tnid]=((double)dz)*NABLA_NB_NODES_Z_TICK;\n":
          "\tnode_coord[tnid]=Real3(((double)dx)*NABLA_NB_NODES_X_TICK,((double)dy)*NABLA_NB_NODES_Y_TICK,((double)dz)*NABLA_NB_NODES_Z_TICK);\n"
          );
} 



/*****************************************************************************
 * Backend CUDA - Allocation de la connectivité du maillage
 *****************************************************************************/
void nccCudaMainMeshPrefix(nablaMain *nabla){
  dbg("\n[nccCudaMainMeshPrefix]");
  fprintf(nabla->entity->src,"\n\n\
\t// Allocation coté CUDA des connectivités aux noeuds\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&node_cell, 8*NABLA_NB_NODES*sizeof(int)));\n\
\t// Allocation coté CUDA des connectivités aux mailles\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&cell_node, 8*NABLA_NB_CELLS*sizeof(int)));\n\
\n\
");
}


void nccCudaMainMeshPostfix(nablaMain *nabla){
  dbg("\n[nccCudaMainMeshPostfix]");
  fprintf(nabla->entity->src,"\n\n\
\tCUDA_HANDLE_ERROR(cudaFree(node_cell));\n\
\tCUDA_HANDLE_ERROR(cudaFree(cell_node));\n\
");
}
