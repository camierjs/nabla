///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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
// * Backend PREINIT - Génération du 'main'
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
\tnabla_ini_connectivity(msh,node_coord,\n\t\t\t\t\t\t\t\t\txs_cell_node,xs_cell_prev,xs_cell_next,xs_cell_face,\n\t\t\t\t\t\t\t\t\txs_node_cell,xs_node_cell_corner,xs_node_cell_and_corner,\n\t\t\t\t\t\t\t\t\txs_face_cell,xs_face_node);\n"
NABLA_STATUS rajaHookMainPreInit(nablaMain *nabla){
  int i;
  dbg("\n[hookMainPreInit]");
  fprintf(nabla->entity->src, "\n\n\t// BACKEND_MAIN_PREINIT");
  if (isWithLibrary(nabla,with_real))
    xHookMesh1DConnectivity(nabla);
  else if (isWithLibrary(nabla,with_real2))
    xHookMesh2DConnectivity(nabla);
  else
    xHookMesh3DConnectivity(nabla,"xs");
  fprintf(nabla->entity->src, BACKEND_MAIN_PREINIT);

  fprintf(nabla->entity->src, "\
\n\t///////////////////////////////\
\n\t// RAJA IndexSet Initialisation\
\n\t///////////////////////////////\
\n\tRAJA::IndexSet *cellIdxSet = new RAJA::IndexSet();\
\n\tRAJA::IndexSet *nodeIdxSet = new RAJA::IndexSet();\
\n\tRAJA::IndexSet *faceIdxSet = new RAJA::IndexSet();\
\n\tcellIdxSet->push_back(RAJA::RangeSegment(0, NABLA_NB_CELLS));\
\n\tnodeIdxSet->push_back(RAJA::RangeSegment(0, NABLA_NB_NODES));\
\n\tfaceIdxSet->push_back(RAJA::RangeSegment(0, NABLA_NB_FACES));\
\n");

  
  nprintf(nabla,NULL,"\n\
\t// ****************************************************************\n\
\t// Initialisation des variables\n\
\t// ****************************************************************");
  // Variables Particulaires
  i=0;
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='p') continue;
    i+=1;
  }
  if (i>0){
    nprintf(nabla,NULL,"/*i=%d*/",i);
    nprintf(nabla,NULL,"\n\tRAJA::forall<particle_exec_policy>(*particleList,[=] RAJA_DEVICE (int p){");
    for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
      if (var->item[0]!='p') continue;
      nprintf(nabla,NULL,"\n\t\t%s_%s[p]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
    }
    nprintf(nabla,NULL,"\n\t});");
  }
  
  // Variables aux noeuds
  i=0;
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='n') continue;
    if (strcmp(var->name, "coord")==0) continue;
    i+=1;
  }
  if (i>0){
    nprintf(nabla,NULL,"\n\tRAJA::forall<node_exec_policy>(*nodeIdxSet,[=] RAJA_DEVICE (int n){");
    for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
      if (var->item[0]!='n') continue;
      if (strcmp(var->name, "coord")==0) continue;
      nprintf(nabla,NULL,"\n\t\t%s_%s[n]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
      if (strcmp(var->type, "real3x3")==0) nprintf(nabla,NULL,"real3x3();");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
    }
    nprintf(nabla,NULL,"\n\t});");
  }
  
  // Variables aux mailles
  i=0;
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='c') continue;
    i+=1;
  }
  if (i>0){
    nprintf(nabla,NULL,"\n\tRAJA::forall<cell_exec_policy>(*cellIdxSet,[=] RAJA_DEVICE (int c){");
    for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
      if (var->item[0]!='c') continue;
      if (var->dim==0){
        nprintf(nabla,NULL,"\n\t\t%s_%s[c]=",var->item,var->name);
        if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
        if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
        if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
        if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
        if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0;");
        if (strcmp(var->type, "real3x3")==0) nprintf(nabla,NULL,"real3x3();");
      }else{
        nprintf(nabla,NULL,"\n\t\tFOR_EACH_CELL_NODE(n)");
        nprintf(nabla,NULL," %s_%s[n+NABLA_NODE_PER_CELL*c]=",var->item,var->name);
        if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"0.0;");
        if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
        if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
        if (strcmp(var->type, "real3x3")==0) nprintf(nabla,NULL,"real3x3();");
        if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
      }
    }
    nprintf(nabla,NULL,"\n\t});");
  }
  return NABLA_OK;
}
