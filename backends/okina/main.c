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

extern char* nablaAlephHeader(nablaMain*);

// ****************************************************************************
// * nOkinaInitVariables
// ****************************************************************************
NABLA_STATUS nOkinaMainVarInitKernel(nablaMain *nabla){
  //int i,iVar;
  nablaVariable *var;
  dbg("\n[nOkinaMainVarInit]");
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
      nprintf(nabla,NULL," %s_%s[n+NABLA_NODE_PER_CELL*c]=",var->item,var->name);
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
// * nOkinaMain
// ****************************************************************************
NABLA_STATUS nOkinaMainHLT(nablaMain *n){
  nablaVariable *var;
  nablaJob *entry_points;
  int i,number_of_entry_points;
  bool is_into_compute_loop=false;
  double last_when;
  
  dbg("\n[nOkinaMain]");
  number_of_entry_points=nMiddleNumberOfEntryPoints(n);
  entry_points=nMiddleEntryPointsSort(n,number_of_entry_points);
  
  // Et on rescan afin de dumper, on rajoute les +2 ComputeLoopEnd|Begin
   for(i=0,last_when=entry_points[i].whens[0];i<number_of_entry_points+2;++i){
     if (strcmp(entry_points[i].name,"ComputeLoopEnd")==0)continue;
     if (strcmp(entry_points[i].name,"ComputeLoopBegin")==0)continue;
     dbg("%s\n\t[nOkinaMain] sorted #%d: %s @ %f in '%s'", (i==0)?"\n":"",i,
        entry_points[i].name,
        entry_points[i].whens[0],
        entry_points[i].where);
    // Si l'on passe pour la première fois la frontière du zéro, on écrit le code pour boucler
    if (entry_points[i].whens[0]>=0 && is_into_compute_loop==false){
      is_into_compute_loop=true;
      nprintf(n, NULL,"\
\n\tgettimeofday(&st, NULL);\n\
\twhile (global_time<option_stoptime){// && global_iteration!=option_max_iterations){");
    }
    // On sync si l'on découvre un temps logique différent
    if (last_when!=entry_points[i].whens[0])
      nprintf(n, NULL, "\n%s%s",
              is_into_compute_loop?"\t\t":"\t",
              n->call->parallel->sync());
    last_when=entry_points[i].whens[0];
    // Dump de la tabulation et du nom du point d'entrée
//#warning ici float, mais ordre!! 
    nprintf(n, NULL, "\n%s/*@%f*/%s%s(",
            is_into_compute_loop?"\t\t":"\t",
            entry_points[i].whens[0],
            n->call->parallel->spawn(), 
            entry_points[i].name);
    
    // Et on dump les in et les out
    if (entry_points[i].nblParamsNode != NULL){
      //nOkinaArgsList(n,entry_points[i].nblParamsNode,&numParams);
    }else nprintf(n,NULL,"/*NULL_nblParamsNode*/");

    // Si on doit appeler des jobs depuis cette fonction @ée
    if (entry_points[i].called_variables != NULL){
      //nOkinaArgsAddExtraConnectivities(n,&numParams);
      // Et on rajoute les called_variables en paramètre d'appel
      dbg("\n\t[nOkinaMain] Et on rajoute les called_variables en paramètre d'appel");
      for(var=entry_points[i].called_variables;var!=NULL;var=var->next){
        nprintf(n, NULL, ",\n\t\t/*used_called_variable*/%s_%s",var->item, var->name);
      }
    }else nprintf(n,NULL,"/*NULL_called_variables*/");
    nprintf(n, NULL, ");");
    //nOkinaArgsDumpNablaDebugFunctionFromOut(n,entry_points[i].nblParamsNode,true);
    //nprintf(n, NULL, "\n");
  }
  return NABLA_OK;
}

