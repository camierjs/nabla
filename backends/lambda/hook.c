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
// * lambdaHookAddArguments
// ****************************************************************************
void lambdaHookAddArguments(nablaMain *nabla,nablaJob *job){
  if (job->parse.function_call_name!=NULL)
    nMiddleArgsDumpFromDFS(nabla, job);
}


// ****************************************************************************
// *
// ****************************************************************************
bool lambdaHookDfsVariable(void){ return true; }


// ****************************************************************************
// * System Prev Cell
// ****************************************************************************
char* lambdaHookPrevCell(int direction){
  if (direction==DIR_X) return "gatherk_and_zero_neg_ones(cell_prev[MD_DirX*NABLA_NB_CELLS+c],";
  if (direction==DIR_Y) return "gatherk_and_zero_neg_ones(cell_prev[MD_DirY*NABLA_NB_CELLS+c],";
  if (direction==DIR_Z) return "gatherk_and_zero_neg_ones(cell_prev[MD_DirZ*NABLA_NB_CELLS+c],";
  assert(NULL);
  return NULL;
}


// ****************************************************************************
// * System Next Cell
// ****************************************************************************
char* lambdaHookNextCell(int direction){
  if (direction==DIR_X) return "gatherk_and_zero_neg_ones(cell_next[MD_DirX*NABLA_NB_CELLS+c],";
  if (direction==DIR_Y) return "gatherk_and_zero_neg_ones(cell_next[MD_DirY*NABLA_NB_CELLS+c],";
  if (direction==DIR_Z) return "gatherk_and_zero_neg_ones(cell_next[MD_DirZ*NABLA_NB_CELLS+c],";
  assert(NULL);
  return NULL;
}


// ****************************************************************************
// * System Postfix
// ****************************************************************************
char* lambdaHookSysPostfix(void){ return ")"; }


// ****************************************************************************
// * Function Hooks
// ****************************************************************************
void lambdaHookFunctionName(nablaMain *nabla){
  nprintf(nabla, NULL, "%s", nabla->name);
}

// ****************************************************************************
// * Dump des variables appelées
// ****************************************************************************
void lambdaHookDfsForCalls(nablaMain *nabla,
                           nablaJob *fct,
                           astNode *n,
                           const char *namespace,
                           astNode *nParams){
  nMiddleFunctionDumpFwdDeclaration(nabla,fct,nParams,namespace);
}


// ****************************************************************************
// * Dump du préfix des points d'entrées: inline ou pas
// ****************************************************************************
char* lambdaHookEntryPointPrefix(nablaMain *nabla,
                                 nablaJob *entry_point){
  return "static inline";
}

void lambdaHookIteration(nablaMain *nabla){
  nprintf(nabla, "/*ITERATION*/", "lambda_iteration()");
}
void lambdaHookExit(nablaMain *nabla, nablaJob *job){
  if (job->when_depth==0)
    nprintf(nabla, "/*EXIT*/", "/*lambdaHookExit*/exit(0.0)");
  else
    nprintf(nabla, "/*EXIT*/", "hlt_exit[hlt_level]=false");
}
void lambdaHookTime(nablaMain *nabla){
  nprintf(nabla, "/*TIME*/", "global_time[0]");
}
void lambdaHookFatal(nablaMain *nabla){
  nprintf(nabla, NULL, "fatal");
}
void lambdaHookAddCallNames(nablaMain *nabla,nablaJob *fct,astNode *n){
  nablaJob *foundJob;
  char *callName=n->next->children->children->token;
  nprintf(nabla, "/*function_got_call*/", "/*%s*/",callName);
  fct->parse.function_call_name=NULL;
  if ((foundJob=nMiddleJobFind(fct->entity->jobs,callName))!=NULL){
    if (foundJob->is_a_function!=true){
      nprintf(nabla, "/*isNablaJob*/", NULL);
      fct->parse.function_call_name=strdup(callName);
    }else{
      nprintf(nabla, "/*isNablaFunction*/", NULL);
    }
  }else{
    nprintf(nabla, "/*has not been found*/", NULL);
  }
}

