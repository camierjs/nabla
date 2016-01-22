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
#include "nabla.tab.h"
#include "frontend/ast.h"

bool lambdaHookDfsVariable(void){ return true; }


// ****************************************************************************
// * INCLUDES
// ****************************************************************************
char *lambdaHookBits(void){return "64";}
char* lambdaHookIncludes(void){return "";}


// ****************************************************************************
// * IVDEP Pragma
// ****************************************************************************
char *lambdaHookPragmaIccIvdep(void){ return "\\\n_Pragma(\"ivdep\")"; }
char *lambdaHookPragmaGccIvdep(void){ return "__declspec(align(64))"; }


// ****************************************************************************
// * ALIGN hooks
// ****************************************************************************
char *lambdaHookPragmaIccAlign(void){ return ""; }
char *lambdaHookPragmaGccAlign(void){ return ""; }


// ****************************************************************************
// * System Prefix
// ****************************************************************************
char* lambdaHookSysPrefix(void){ return "/*lambdaHookSysPrefix*/"; }


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
char* lambdaHookSysPostfix(void){ return "/*lambdaHookSysPostfix*/)"; }


// ****************************************************************************
// * Function Hooks
// ****************************************************************************
void lambdaHookFunctionName(nablaMain *arc){
  nprintf(arc, NULL, "%s", arc->name);
}
void lambdaHookFunction(nablaMain *nabla, astNode *n){
  nablaJob *fct=nMiddleJobNew(nabla->entity);
  nMiddleJobAdd(nabla->entity, fct);
  nMiddleFunctionFill(nabla,fct,n,NULL);
}


// ****************************************************************************
// * Dump des variables appelées
// ****************************************************************************
void lambdaHookDfsForCalls(struct nablaMainStruct *nabla,
                             nablaJob *fct,
                             astNode *n,
                             const char *namespace,
                             astNode *nParams){
  //nMiddleDfsForCalls(nabla,fct,n,namespace,nParams);
  nMiddleFunctionDumpFwdDeclaration(nabla,fct,nParams,namespace);
}


// ****************************************************************************
// * Dump du préfix des points d'entrées: inline ou pas
// ****************************************************************************
char* lambdaHookEntryPointPrefix(struct nablaMainStruct *nabla, nablaJob *entry_point){
  //return "";
  return "static inline";
}

void lambdaHookIteration(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*ITERATION*/", "lambda_iteration()");
}
void lambdaHookExit(struct nablaMainStruct *nabla, nablaJob *job){
  if (job->when_depth==0)
    nprintf(nabla, "/*EXIT*/", "/*lambdaHookExit*/exit(0.0)");
  else
    nprintf(nabla, "/*EXIT*/", "hlt_exit[hlt_level]=false");
}
void lambdaHookTime(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*TIME*/", "global_time[0]");
}
void lambdaHookFatal(struct nablaMainStruct *nabla){
  nprintf(nabla, NULL, "fatal");
}
void lambdaHookAddCallNames(struct nablaMainStruct *nabla,nablaJob *fct,astNode *n){
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


/*****************************************************************************
 * Lambda libraries
 *****************************************************************************/
void lambdaHookLibraries(astNode * n, nablaEntity *entity){
  fprintf(entity->src, "\n/*lib %s*/",n->children->token);
}


// ****************************************************************************
// * lambdaHookPrimaryExpressionToReturn
// ****************************************************************************
bool lambdaHookPrimaryExpressionToReturn(nablaMain *nabla, nablaJob *job, astNode *n){
  const char* var=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
  dbg("\n\t[lambdaHookPrimaryExpressionToReturn] ?");
  if (var!=NULL && strcmp(n->children->token,var)==0){
    dbg("\n\t[lambdaHookPrimaryExpressionToReturn] primaryExpression hits returned argument");
    nprintf(nabla, NULL, "%s_per_thread[tid]",var);
    return true;
  }else{
    dbg("\n\t[lambdaHookPrimaryExpressionToReturn] ELSE");
    //nprintf(nabla, NULL, "%s",n->children->token);
  }
  return false;
}



/*****************************************************************************
 * Génération d'un kernel associé à un support
 *****************************************************************************/
void lambdaHookJob(nablaMain *nabla, astNode *n){
  nablaJob *job = nMiddleJobNew(nabla->entity);
  nMiddleJobAdd(nabla->entity, job);
  nMiddleJobFill(nabla,job,n,NULL);
  
  // On teste *ou pas* que le job retourne bien 'void' dans le cas de LAMBDA
  //if ((strcmp(job->rtntp,"void")!=0) && (job->is_an_entry_point==true))
  //  exit(NABLA_ERROR|fprintf(stderr, "\n[lambdaHookJob] Error with return type which is not void\n"));
}


