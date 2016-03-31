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

//extern char* nablaAlephHeader(nablaMain*);

// ****************************************************************************
// * kHookMainPrefix
// ****************************************************************************
#define BACKEND_MAIN_PREFIX "\n\
// ******************************************************************************\n\
// * Main\n\
// ******************************************************************************\n\
int main(int argc, char *argv[]){\n\
\tKokkos::initialize(argc, argv);\n\
\tprintf(\"Kokkos execution space %%s\\n\",\n\
\t\ttypeid(Kokkos::DefaultExecutionSpace).name ());\n\
\tfloat alltime=0.0;\n\
\tstruct timeval st, et;\n\
\t__attribute__((unused)) int NABLA_NB_PARTICLES;\n\
\tif (argc==1)\n\
\t\tNABLA_NB_PARTICLES=1000;\n\
\telse\n\
\t\tNABLA_NB_PARTICLES=atoi(argv[1]);\n\
\t// Initialisation des swirls\n\
\tint hlt_level=0;\n\
\tbool* hlt_exit=(bool*)calloc(64,sizeof(bool));\n\
\t// Initialisation de la précision du cout\n\
\tstd::cout.precision(14);//21, 14 pour Arcane\n\
\t//std::cout.setf(std::ios::floatfield);\n\
\tstd::cout.setf(std::ios::scientific, std::ios::floatfield);"
NABLA_STATUS kHookMainPrefix(nablaMain *nabla){
  //if ((nabla->entity->libraries&(1<<with_aleph))!=0)
  //fprintf(nabla->entity->hdr, "%s", nablaAlephHeader(nabla));
  fprintf(nabla->entity->src, BACKEND_MAIN_PREFIX);
  return NABLA_OK;
}


// ****************************************************************************
// * kHookMainPreInit
// ****************************************************************************
#define BACKEND_MAIN_PREINIT "\
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
\t\tNABLA_NB_CELLS_WARP};\n\
\tprintf(\"%%d noeuds, %%d mailles & %%d faces\",NABLA_NB_NODES,NABLA_NB_CELLS,NABLA_NB_FACES);\n \
\tnabla_ini_connectivity(msh,\n\t\t\t\t\t\t\t\t\tnode_coord.ptr_on_device(),\n\t\t\t\t\t\t\t\t\txs_cell_node.ptr_on_device(),\n\t\t\t\t\t\t\t\t\txs_cell_prev.ptr_on_device(),\n\t\t\t\t\t\t\t\t\txs_cell_next.ptr_on_device(),\n\t\t\t\t\t\t\t\t\txs_cell_face.ptr_on_device(),\n\t\t\t\t\t\t\t\t\txs_node_cell.ptr_on_device(),\n\t\t\t\t\t\t\t\t\txs_node_cell_corner.ptr_on_device(),\n\t\t\t\t\t\t\t\t\txs_node_cell_and_corner.ptr_on_device(),\n\t\t\t\t\t\t\t\t\txs_face_cell.ptr_on_device(),\n\t\t\t\t\t\t\t\t\txs_face_node.ptr_on_device());\n"
NABLA_STATUS kHookMainPreInit(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\
\t// ********************************************************\n\
\t// Initialisation du temps et du deltaT\n\
\t// ********************************************************\n\
\tglobal_time[0]=option_dtt_initial;\n\
\tglobal_iteration[0]=1;\n\
\tglobal_deltat[0]=option_dtt_initial;\n\n\t// kHookMainPreInit\n\
\t// ********************************************************\n\
\t// * MESH CONNECTIVITY (3D) with prefix 'xs'\n\
\t// ********************************************************\n\
\tKokkos::View<int*> xs_cell_node(\"xs_cell_node_label\",NABLA_NB_CELLS*NABLA_NODE_PER_CELL);\n\
\tKokkos::View<int*> xs_cell_next(\"xs_cell_next_label\",NABLA_NB_CELLS*3);\n\
\tKokkos::View<int*> xs_cell_prev(\"xs_cell_prev_label\",NABLA_NB_CELLS*3);\n\
\tKokkos::View<int*> xs_cell_face(\"xs_cell_face_label\",NABLA_NB_CELLS*NABLA_FACE_PER_CELL);\n\
\tKokkos::View<int*> xs_node_cell(\"xs_node_cell_label\",NABLA_NB_NODES*NABLA_CELL_PER_NODE);\n\
\tKokkos::View<int*> xs_node_cell_corner(\"xs_node_cell_corner_label\",NABLA_NB_NODES*NABLA_CELL_PER_NODE);\n\
\tKokkos::View<int*> xs_node_cell_and_corner(\"xs_node_cell_and_corner_label\",NABLA_NB_NODES*2*NABLA_CELL_PER_NODE);\n\
\tKokkos::View<int*> xs_face_cell(\"xs_face_cell_label\",NABLA_NB_FACES*NABLA_CELL_PER_FACE);\n\
\tKokkos::View<int*> xs_face_node(\"xs_face_node_label\",NABLA_NB_FACES*NABLA_NODE_PER_FACE);\n");
  fprintf(nabla->entity->src, BACKEND_MAIN_PREINIT);
  return NABLA_OK;
}


// ****************************************************************************
// * kHookMainPostfix
// ****************************************************************************
#define BACKEND_MAIN_POSTFIX "\n\t\t//BACKEND_MAIN_POSTFIX\
\n\t\tglobal_time[0]+=global_deltat[0];\
\n\t\tglobal_iteration[0]+=1;\
\n\t\t//printf(\"\\ttime=%%e, dt=%%e\\n\", global_time[0], global_deltat[0]);\
\n\t}\
\n\tgettimeofday(&et, NULL);\n\
\talltime = ((et.tv_sec-st.tv_sec)*1000.+ (et.tv_usec - st.tv_usec)/1000.0);\n\
\tprintf(\"\\n\\t\\33[7m[#%%04d] Elapsed time = %%12.6e(s)\\33[m\\n\", global_iteration[0]-1, alltime/1000.0);\n"
NABLA_STATUS kHookMainPostfix(nablaMain *nabla){
  fprintf(nabla->entity->src, BACKEND_MAIN_POSTFIX);
  fprintf(nabla->entity->src, "\n\t#warning no Kokkos::finalize();");
  return NABLA_OK;
}
