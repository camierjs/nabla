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


// ****************************************************************************
// * nccOkinaMainVarInitKernel
// ****************************************************************************
NABLA_STATUS nccOkinaMainVarInitKernel(nablaMain *nabla){
  //int i,iVar;
  nablaVariable *var;
  dbg("\n[nccOkinaMainVarInit]");
  nprintf(nabla,NULL,"\n\
// ******************************************************************************\n\
// * Kernel d'initialisation des variables\n\
// ******************************************************************************\n\
void nabla_ini_variables(void){");
  // Variables aux noeuds
  nprintf(nabla,NULL,"\n\tFOR_EACH_NODE_WARP(n){");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='n') continue;
    if (strcmp(var->name, "coord")==0) continue;
    nprintf(nabla,NULL,"\n\t\t%s_%s[n]=",var->item,var->name);
    if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
    if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
    if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0;");
  }
  nprintf(nabla,NULL,"\n\t}");  
  // Variables aux mailles real
  nprintf(nabla,NULL,"\n\tFOR_EACH_CELL_WARP(c){");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='c') continue;
    if (var->dim==0){
      nprintf(nabla,NULL,"\n\t\t%s_%s[c]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0;");
    }else{
      nprintf(nabla,NULL,"\n\t\tFOR_EACH_CELL_WARP_NODE(n)");
      nprintf(nabla,NULL," %s_%s[n+8*c]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"0.0;");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0;");
    }
  }
  nprintf(nabla,NULL,"\n\t}");
  nprintf(nabla,NULL,"\n}");
  return NABLA_OK;
}


// ****************************************************************************
// * nccOkinaMainVarInitKernel
// ****************************************************************************
NABLA_STATUS nccOkinaMainVarInitCall(nablaMain *nabla){
  nablaVariable *var;
  dbg("\n[nccOkinaMainVarInitCall]");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (strcmp(var->name, "deltat")==0) continue;
    if (strcmp(var->name, "time")==0) continue;
    if (strcmp(var->name, "coord")==0) continue;
    nprintf(nabla,NULL,"\n\t//printf(\"\\ndbgsVariable %s\"); dbg%sVariable%sDim%s_%s();",
            var->name,
            (var->item[0]=='n')?"Node":"Cell",
            (strcmp(var->type,"real3")==0)?"XYZ":"",
            (var->dim==0)?"0":"1",
            var->name);
  }
  return NABLA_OK;
}
