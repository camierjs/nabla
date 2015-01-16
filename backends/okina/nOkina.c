// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
#include "nabla.h"
#include "nabla.tab.h"
#include "frontend/nablaAst.h"


// ****************************************************************************
// * Dump des variables appelées
// ****************************************************************************
static void okinaDfsForCalls(struct nablaMainStruct *nabla,
                             nablaJob *fct,
                             astNode *n,
                             const char *namespace,
                             astNode *nParams){
  // Maintenant qu'on a tous les called_variables potentielles, on remplit aussi le hdr
  // On remplit la ligne du hdr
  hprintf(nabla, NULL, "\n%s %s %s%s(",
          nabla->hook->entryPointPrefix(nabla,fct),
          fct->rtntp,
          namespace?"Entity::":"",
          fct->name);
  // On va chercher les paramètres standards pour le hdr
  dumpParameterTypeList(nabla->entity->hdr, nParams);
  hprintf(nabla, NULL, ");");
}


// ****************************************************************************
// * Dump du préfix des points d'entrées: inline ou pas
// ****************************************************************************
static char* okinaEntryPointPrefix(struct nablaMainStruct *nabla, nablaJob *entry_point){
  //return "";
  return "static inline";
}

static void okinaIteration(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*ITERATION*/", "okina_iteration()");
}
static void okinaExit(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*EXIT*/", "exit(0.0)");
}
static void okinaTime(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*TIME*/", "global_time");
}
static void okinaFatal(struct nablaMainStruct *nabla){
  nprintf(nabla, NULL, "fatal");
}
static void okinaAddCallNames(struct nablaMainStruct *nabla,nablaJob *fct,astNode *n){
  nablaJob *foundJob;
  char *callName=n->next->children->children->token;
  nprintf(nabla, "/*function_got_call*/", "/*%s*/",callName);
  fct->parse.function_call_name=NULL;
  if ((foundJob=nablaJobFind(fct->entity->jobs,callName))!=NULL){
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
static void okinaAddArguments(struct nablaMainStruct *nabla,nablaJob *fct){
  // En Okina, par contre il faut les y mettre
  if (fct->parse.function_call_name!=NULL){
    //nprintf(nabla, "/*ShouldDumpParamsInOkina*/", "/*Arg*/");
    int numParams=1;
    nablaJob *called=nablaJobFind(fct->entity->jobs,fct->parse.function_call_name);
    okinaAddExtraArguments(nabla, called, &numParams);
    if (called->nblParamsNode != NULL)
      okinaDumpNablaArgumentList(nabla,called->nblParamsNode,&numParams);
  }
}

static void okinaTurnTokenToOption(struct nablaMainStruct *nabla,nablaOption *opt){
  nprintf(nabla, "/*tt2o okina*/", "%s", opt->name);
}



/*****************************************************************************
 * Okina libraries
 *****************************************************************************/
void okinaHookLibraries(astNode * n, nablaEntity *entity){
  fprintf(entity->src, "\n/*lib %s*/",n->children->token);
}


// ****************************************************************************
// * okinaInclude
// ****************************************************************************
void okinaInclude(nablaMain *nabla){
  fprintf(nabla->entity->src,"#include \"%s.h\"\n", nabla->entity->name);
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void okinaHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,
          "#ifndef __OKINA_%s_H__\n#define __OKINA_%s_H__",
          nabla->entity->name,nabla->entity->name);
}

/***************************************************************************** 
 * 
 *****************************************************************************/
static void okinaHeaderIncludes(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,"\n\n\n\
// *****************************************************************************\n\
// * Okina includes\n\
// *****************************************************************************\n\
%s // from nabla->simd->includes\n\
#include <sys/time.h>\n\
#include <stdlib.h>\n\
#include <stdio.h>\n\
#include <string.h>\n\
#include <vector>\n\
#include <math.h>\n\
#include <assert.h>\n\
#include <stdarg.h>\n\
//#include <mathimf.h>\n\
#include <iostream>\n\
%s // fromnabla->parallel->includes()\n\
%s // SOA define or not",
          nabla->simd->includes(),
          nabla->parallel->includes(),
          ((nabla->colors&BACKEND_COLOR_OKINA_SOA)==BACKEND_COLOR_OKINA_SOA)?"#define _OKINA_SOA_":"");
}


// ****************************************************************************
// * okinaHeader for Std, Avx or Mic
// ****************************************************************************
extern char knStdReal_h[];
extern char knStdReal3_h[];
extern char knStdInteger_h[];
extern char knStdGather_h[];
extern char knStdScatter_h[];
extern char knStdOStream_h[];
extern char knStdTernary_h[];

extern char knSseReal_h[];
extern char knSseReal3_h[];
extern char knSseInteger_h[];
extern char knSseGather_h[];
extern char knSseScatter_h[];
extern char knSseOStream_h[];
extern char knSseTernary_h[];

extern char knAvxReal_h[];
extern char knAvxReal3_h[];
extern char knAvxInteger_h[];
extern char knAvxGather_h[];
extern char knAvx2Gather_h[];
extern char knAvxScatter_h[];
extern char knAvxOStream_h[];
extern char knAvxTernary_h[];

extern char knMicReal_h[];
extern char knMicReal3_h[];
extern char knMicInteger_h[];
extern char knMicGather_h[];
extern char knMicScatter_h[];
extern char knMicOStream_h[];
extern char knMicTernary_h[];
static char *dumpExternalFile(char *file){
  return file+NABLA_GPL_HEADER;
}
static void okinaHeaderSimd(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC){
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicInteger_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicReal_h));
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicReal3_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicTernary_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicGather_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicScatter_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicOStream_h));
  }else if (((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)||
            ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)){
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxInteger_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxReal_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxReal3_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxTernary_h));
    if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
      fprintf(nabla->entity->hdr,dumpExternalFile(knAvxGather_h));
    else
      fprintf(nabla->entity->hdr,dumpExternalFile(knAvx2Gather_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxScatter_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxOStream_h));
  }else if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE){
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseInteger_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseReal_h));
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseReal3_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseTernary_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseGather_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseScatter_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseOStream_h));
  }else{
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdInteger_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdReal_h));
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdReal3_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdTernary_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdGather_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdScatter_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdOStream_h));
  }
}


// ****************************************************************************
// * okinaHeader for Dbg
// ****************************************************************************
extern char knDbg_h[];
static void okinaHeaderDbg(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(knDbg_h));
}


// ****************************************************************************
// * okinaHeader for Maths
// ****************************************************************************
extern char knMth_h[];
static void okinaHeaderMth(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(knMth_h));
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void okinaHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n#endif // __OKINA_%s_H__\n",nabla->entity->name);
}


// ****************************************************************************
// * okinaPrimaryExpressionToReturn
// ****************************************************************************
static bool okinaPrimaryExpressionToReturn(nablaMain *nabla, nablaJob *job, astNode *n){
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
static void okinaReturnFromArgument(nablaMain *nabla, nablaJob *job){
  const char *rtnVariable=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
  if ((nabla->colors&BACKEND_COLOR_OKINA_OpenMP)==BACKEND_COLOR_OKINA_OpenMP)
    nprintf(nabla, NULL, "\
\n\tint threads = omp_get_max_threads();\
\n\tReal %s_per_thread[threads];", rtnVariable);
}


// ****************************************************************************
// * okinaHookReduction
// ****************************************************************************
static void okinaHookReduction(struct nablaMainStruct *nabla, astNode *n){
  const astNode *item_node = n->children->next->children;
  const astNode *global_var_node = n->children->next->next;
  const astNode *reduction_operation_node = global_var_node->next;
  const astNode *item_var_node = reduction_operation_node->next;
  const astNode *at_single_cst_node = item_var_node->next->next->children->next->children;
  char *global_var_name = global_var_node->token;
  char *item_var_name = item_var_node->token;
  // Préparation du nom du job
  char job_name[NABLA_MAX_FILE_NAME];
  job_name[0]=0;
  strcat(job_name,"okinaReduction_");
  strcat(job_name,global_var_name);
  // Rajout du job de reduction
  nablaJob *redjob = nablaJobNew(nabla->entity);
  redjob->is_an_entry_point=true;
  redjob->is_a_function=false;
  redjob->scope  = strdup("NoGroup");
  redjob->region = strdup("NoRegion");
  redjob->item   = strdup(item_node->token);
  redjob->rtntp  = strdup("void");
  redjob->name   = strdup(job_name);
  redjob->name_utf8 = strdup(job_name);
  redjob->xyz    = strdup("NoXYZ");
  redjob->drctn  = strdup("NoDirection");
  assert(at_single_cst_node->parent->ruleid==rulenameToId("at_single_constant"));
  dbg("\n\t[okinaHookReduction] @ %s",at_single_cst_node->token);
  sprintf(&redjob->at[0],at_single_cst_node->token);
  redjob->whenx  = 1;
  redjob->whens[0] = atof(at_single_cst_node->token);
  nablaJobAdd(nabla->entity, redjob);
  const double reduction_init = (reduction_operation_node->tokenid==MIN_ASSIGN)?1.0e20:0.0;
  // Génération de code associé à ce job de réduction
  nprintf(nabla, NULL, "\n\
// ******************************************************************************\n\
// * Kernel de reduction de la variable '%s' vers la globale '%s'\n\
// ******************************************************************************\n\
void %s(void){ // @ %s\n\
\tconst double reduction_init=%e;\n\
\tconst int threads = omp_get_max_threads();\n\
\tReal %s_per_thread[threads];\n\
\tdbgFuncIn();\n\
\tfor (int i=0; i<threads;i+=1) %s_per_thread[i] = reduction_init;\n\
\tFOR_EACH_%s_WARP_SHARED(%s,reduction_init){\n\
\t\tint tid = omp_get_thread_num();\n\
\t\t%s_per_thread[tid] = min(%s_%s[%s],%s_per_thread[tid]);\n\
\t}\n\
\tfor (int i=0; i<threads; i+=1){\n\
\t\tglobal_%s[0]=(ReduceMinToDouble(%s_per_thread[i])<reduction_init)?\n\
\t\t\t\t\t\t\t\t\tReduceMinToDouble(%s_per_thread[i]):reduction_init;\n\
\t}\n\
}\n\n",   item_var_name,global_var_name,
          job_name,
          at_single_cst_node->token,
          reduction_init,
          global_var_name,
          global_var_name,
          (item_node->token[0]=='c')?"CELL":(item_node->token[0]=='n')?"NODE":"NULL",
          (item_node->token[0]=='c')?"c":(item_node->token[0]=='n')?"n":"?",
          global_var_name,
          (item_node->token[0]=='c')?"cell":(item_node->token[0]=='n')?"node":"?",
          item_var_name,
          (item_node->token[0]=='c')?"c":(item_node->token[0]=='n')?"n":"?",
          global_var_name,
          global_var_name,global_var_name,global_var_name
          );  
}


/*****************************************************************************
 * nccOkina
 *****************************************************************************/
NABLA_STATUS nccOkina(nablaMain *nabla,
                      astNode *root,
                      const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  // Définition des hooks pour l'AVX ou le MIC
  nablaBackendSimdHooks nablaOkinaSimdStdHooks={
    okinaStdBits,
    okinaStdGather,
    okinaStdScatter,
    okinaStdTypedef,
    okinaStdDefines,
    okinaStdForwards,
    okinaStdPrevCell,
    okinaStdNextCell,
    okinaStdIncludes
  };
  nablaBackendSimdHooks nablaOkinaSimdSseHooks={
    okinaSseBits,
    okinaSseGather,
    okinaSseScatter,
    okinaSseTypedef,
    okinaSseDefines,
    okinaSseForwards,
    okinaSsePrevCell,
    okinaSseNextCell,
    okinaSseIncludes
  };
  nablaBackendSimdHooks nablaOkinaSimdAvxHooks={
    okinaAvxBits,
    okinaAvxGather,
    okinaAvxScatter,
    okinaAvxTypedef,
    okinaAvxDefines,
    okinaAvxForwards,
    okinaAvxPrevCell,
    okinaAvxNextCell,
    okinaAvxIncludes
  };
  nablaBackendSimdHooks nablaOkinaSimdMicHooks={ 
    okinaMicBits,
    okinaMicGather,
    okinaMicScatter,
    okinaMicTypedef,
    okinaMicDefines,
    okinaMicForwards,
    okinaMicPrevCell,
    okinaMicNextCell,
    okinaMicIncludes
  };

  // Switching between STD, SSE, AVX, MIC
  nabla->simd=&nablaOkinaSimdStdHooks;
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
    nabla->simd=&nablaOkinaSimdSseHooks;
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
    nabla->simd=&nablaOkinaSimdAvxHooks;
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
    nabla->simd=&nablaOkinaSimdAvxHooks;
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
    nabla->simd=&nablaOkinaSimdMicHooks;
    
  // Définition des hooks pour Cilk+ *ou pas*
  nablaBackendParallelHooks okinaCilkHooks={
    nccOkinaParallelCilkSync,
    nccOkinaParallelCilkSpawn,
    nccOkinaParallelCilkLoop,
    nccOkinaParallelCilkIncludes
  };
  nablaBackendParallelHooks okinaOpenMPHooks={
    nccOkinaParallelOpenMPSync,
    nccOkinaParallelOpenMPSpawn,
    nccOkinaParallelOpenMPLoop,
    nccOkinaParallelOpenMPIncludes
  };
  nablaBackendParallelHooks okinaVoidHooks={
    nccOkinaParallelVoidSync,
    nccOkinaParallelVoidSpawn,
    nccOkinaParallelVoidLoop,
    nccOkinaParallelVoidIncludes
  };
  nabla->parallel=&okinaVoidHooks;
  if ((nabla->colors&BACKEND_COLOR_OKINA_CILK)==BACKEND_COLOR_OKINA_CILK)
    nabla->parallel=&okinaCilkHooks;
  if ((nabla->colors&BACKEND_COLOR_OKINA_OpenMP)==BACKEND_COLOR_OKINA_OpenMP)
    nabla->parallel=&okinaOpenMPHooks;

  
  nablaBackendPragmaHooks okinaPragmaICCHooks ={
    nccOkinaPragmaIccIvdep,
    nccOkinaPragmaIccAlign
  };
  nablaBackendPragmaHooks okinaPragmaGCCHooks={
    nccOkinaPragmaGccIvdep,
    nccOkinaPragmaGccAlign
  };
  // Par defaut, on met GCC
  nabla->pragma=&okinaPragmaGCCHooks;
  if ((nabla->colors&BACKEND_COLOR_OKINA_ICC)==BACKEND_COLOR_OKINA_ICC)
    nabla->pragma=&okinaPragmaICCHooks;
  
  static nablaBackendHooks okinaBackendHooks={
    // Jobs stuff
    okinaHookPrefixEnumerate,
    okinaHookDumpEnumerateXYZ,
    okinaHookDumpEnumerate,
    okinaHookPostfixEnumerate,
    okinaHookItem,
    okinaHookSwitchToken,
    okinaHookTurnTokenToVariable,
    okinaHookSystem,
    okinaHookAddExtraParameters,
    okinaHookDumpNablaParameterList,
    okinaHookTurnBracketsToParentheses,
    okinaHookJobDiffractStatement,
    // Other hooks
    okinaHookFunctionName,
    okinaHookFunction,
    okinaHookJob,
    okinaHookReduction,
    okinaIteration,
    okinaExit,
    okinaTime,
    okinaFatal,
    okinaAddCallNames,
    okinaAddArguments,
    okinaTurnTokenToOption,
    okinaEntryPointPrefix,
    okinaDfsForCalls,
    okinaPrimaryExpressionToReturn,
    okinaReturnFromArgument
  };
  nabla->hook=&okinaBackendHooks;

  // Rajout de la variable globale 'iteration'
  nablaVariable *iteration = nablaVariableNew(nabla);
  nablaVariableAdd(nabla, iteration);
  iteration->axl_it=false;
  iteration->item=strdup("global");
  iteration->type=strdup("integer");
  iteration->name=strdup("iteration");
 
  // Ouverture du fichier source du entity
  sprintf(srcFileName, "%s.cc", nabla->name);
  if ((nabla->entity->src=fopen(srcFileName, "w")) == NULL) exit(NABLA_ERROR);

  // Ouverture du fichier header du entity
  sprintf(hdrFileName, "%s.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
  
  // Dump dans le HEADER des includes, typedefs, defines, debug, maths & errors stuff
  okinaHeaderPrefix(nabla);
  okinaHeaderIncludes(nabla);
  nablaDefines(nabla,nabla->simd->defines);
  nablaTypedefs(nabla,nabla->simd->typedefs);
  nablaForwards(nabla,nabla->simd->forwards);

  // On inclue les fichiers kn'SIMD'
  okinaHeaderSimd(nabla);
  okinaHeaderDbg(nabla);
  okinaHeaderMth(nabla);
  okinaMesh(nabla);
  okinaDefineEnumerates(nabla);

  // Dump dans le fichier SOURCE
  okinaInclude(nabla);
  
  // Parse du code préprocessé et lance les hooks associés
  nablaMiddlendParseAndHook(root,nabla);
  nccOkinaMainVarInitKernel(nabla);

  // Partie PREFIX
  nccOkinaMainPrefix(nabla);
  okinaVariablesPrefix(nabla);
  nccOkinaMainMeshPrefix(nabla);
  
  // Partie Pré Init
  nccOkinaMainPreInit(nabla);
  nccOkinaMainVarInitCall(nabla);
      
  // Dump des entry points dans le main
  nccOkinaMain(nabla);

  // Partie Post Init
  nccOkinaMainPostInit(nabla);
  
  // Partie POSTFIX
  okinaHeaderPostfix(nabla); 
  nccOkinaMainMeshPostfix(nabla);
  okinaVariablesPostfix(nabla);
  nccOkinaMainPostfix(nabla);
  return NABLA_OK;
}
