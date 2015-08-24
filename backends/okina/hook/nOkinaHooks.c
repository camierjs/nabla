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
#include "nabla.tab.h"


// ****************************************************************************
// * Dump des variables appelées
// ****************************************************************************
void nOkinaHookDfsForCalls(struct nablaMainStruct *nabla,
                                 nablaJob *fct,
                                 astNode *n,
                                 const char *namespace,
                                 astNode *nParams){
  // Maintenant qu'on a tous les called_variables potentielles, on remplit aussi le hdr
  // On remplit la ligne du hdr
  hprintf(nabla, NULL, "\n%s %s %s%s(",
          nabla->hook->call->entryPointPrefix(nabla,fct),
          fct->return_type,
          namespace?"Entity::":"",
          fct->name);
  // On va chercher les paramètres standards pour le hdr
  nMiddleDumpParameterTypeList(nabla->entity->hdr, nParams);
  hprintf(nabla, NULL, ");");
}


// ****************************************************************************
// * Dump du préfix des points d'entrées: inline ou pas
// ****************************************************************************
char* nOkinaHookEntryPointPrefix(struct nablaMainStruct *nabla, nablaJob *entry_point){
  //return "";
  return "static inline";
}


// ****************************************************************************
// * nOkinaHookIteration
// ****************************************************************************
void nOkinaHookIteration(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*ITERATION*/", "okina_iteration()");
}


// ****************************************************************************
// * nOkinaHookExit
// ****************************************************************************
void nOkinaHookExit(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*EXIT*/", "exit(0.0)");
}


// ****************************************************************************
// * nOkinaHookTime
// ****************************************************************************
void nOkinaHookTime(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*TIME*/", "global_time");
}


// ****************************************************************************
// * nOkinaHookFatal
// ****************************************************************************
void nOkinaHookFatal(struct nablaMainStruct *nabla){
  nprintf(nabla, NULL, "fatal");
}


// ****************************************************************************
// * nOkinaHookAddCallNames
// ****************************************************************************
void nOkinaHookAddCallNames(struct nablaMainStruct *nabla,nablaJob *fct,astNode *n){
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


// ****************************************************************************
// * nOkinaHookAddArguments
// ****************************************************************************
void nOkinaHookAddArguments(struct nablaMainStruct *nabla,nablaJob *fct){
  // En Okina, par contre il faut les y mettre
  if (fct->parse.function_call_name!=NULL){
    //nprintf(nabla, "/*ShouldDumpParamsInOkina*/", "/*Arg*/");
    int numParams=1;
    nablaJob *called=nMiddleJobFind(fct->entity->jobs,fct->parse.function_call_name);
    nOkinaArgsExtra(nabla, called, &numParams);
    if (called->nblParamsNode != NULL)
      nOkinaArgsList(nabla,called->nblParamsNode,&numParams);
  }
}


// ****************************************************************************
// * nOkinaHookTurnTokenToOption
// ****************************************************************************
void nOkinaHookTurnTokenToOption(struct nablaMainStruct *nabla,nablaOption *opt){
  nprintf(nabla, "/*tt2o okina*/", "%s", opt->name);
}


// ****************************************************************************
// * Okina libraries
// ****************************************************************************
void nOkinaHookLibraries(astNode * n, nablaEntity *entity){
  fprintf(entity->src, "\n/*lib %s*/",n->children->token);
}


// ****************************************************************************
// * Génération d'un kernel associé à un support
// ****************************************************************************
void nOkinaHookJob(nablaMain *nabla, astNode *n){
  nablaJob *job = nMiddleJobNew(nabla->entity);
  nMiddleJobAdd(nabla->entity, job);
  nMiddleJobFill(nabla,job,n,NULL);
  // On teste *ou pas* que le job retourne bien 'void' dans le cas de OKINA
  //if ((strcmp(job->return_type,"void")!=0) && (job->is_an_entry_point==true))
  //  exit(NABLA_ERROR|fprintf(stderr, "\n[nOkinaHookJob] Error with return type which is not void\n"));
}




// ****************************************************************************
// * okinaPrimaryExpressionToReturn
// ****************************************************************************
bool nOkinaHookPrimaryExpressionToReturn(nablaMain *nabla, nablaJob *job, astNode *n){
  const char* var=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
  dbg("\n\t[okinaPrimaryExpressionToReturn] ?");
  if (var!=NULL && strcmp(n->children->token,var)==0){
    dbg("\n\t[okinaPrimaryExpressionToReturn] primaryExpression hits returned argument");
    nprintf(nabla, NULL, "%s_per_thread[tid]",var);
    return true;
  }else{
    dbg("\n\t[okinaPrimaryExpressionToReturn] ELSE");
    //nprintf(nabla, NULL, "%s",n->children->token);
  }
  return false;
}


// ****************************************************************************
// * okinaReturnFromArgument
// ****************************************************************************
void nOkinaHookReturnFromArgument(nablaMain *nabla, nablaJob *job){
  const char *rtnVariable=dfsFetchFirst(job->stdParamsNode,
                                        rulenameToId("direct_declarator"));
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nprintf(nabla, NULL, "\
\n\tint threads = omp_get_max_threads();\
\n\tReal %s_per_thread[threads];", rtnVariable);
}


