/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM
 *****************************************************************************
 * File     : nccOkina.c
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
    nprintf(nabla, "/*ShouldDumpParamsInOkina*/", "/*Arg*/");
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
static void okinaHeaderSimd(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC){
    fprintf(nabla->entity->hdr,knMicInteger_h);
    fprintf(nabla->entity->hdr,knMicReal_h);
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
      fprintf(nabla->entity->hdr,knMicReal3_h);
    fprintf(nabla->entity->hdr,knMicTernary_h);
    fprintf(nabla->entity->hdr,knMicGather_h);
    fprintf(nabla->entity->hdr,knMicScatter_h);
    fprintf(nabla->entity->hdr,knMicOStream_h);
  }else if (((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)||
            ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)){
    fprintf(nabla->entity->hdr,knAvxInteger_h);
    fprintf(nabla->entity->hdr,knAvxReal_h);
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
      fprintf(nabla->entity->hdr,knAvxReal3_h);
    fprintf(nabla->entity->hdr,knAvxTernary_h);
    fprintf(nabla->entity->hdr,knAvxGather_h);
    fprintf(nabla->entity->hdr,knAvxScatter_h);
    fprintf(nabla->entity->hdr,knAvxOStream_h);
  }else if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE){
    fprintf(nabla->entity->hdr,knSseInteger_h);
    fprintf(nabla->entity->hdr,knSseReal_h);
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
      fprintf(nabla->entity->hdr,knSseReal3_h);
    fprintf(nabla->entity->hdr,knSseTernary_h);
    fprintf(nabla->entity->hdr,knSseGather_h);
    fprintf(nabla->entity->hdr,knSseScatter_h);
    fprintf(nabla->entity->hdr,knSseOStream_h);
  }else{
    fprintf(nabla->entity->hdr,knStdInteger_h);
    fprintf(nabla->entity->hdr,knStdReal_h);
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
      fprintf(nabla->entity->hdr,knStdReal3_h);
    fprintf(nabla->entity->hdr,knStdTernary_h);
    fprintf(nabla->entity->hdr,knStdGather_h);
    fprintf(nabla->entity->hdr,knStdScatter_h);
    fprintf(nabla->entity->hdr,knStdOStream_h);
  }
}


// ****************************************************************************
// * okinaHeader for Dbg
// ****************************************************************************
extern char knDbg_h[];
static void okinaHeaderDbg(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,knDbg_h);
}


// ****************************************************************************
// * okinaHeader for Maths
// ****************************************************************************
extern char knMth_h[];
static void okinaHeaderMth(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,knMth_h);
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
static void okinaPrimaryExpressionToReturn(nablaMain *nabla, nablaJob *job, astNode *n){
  const char* var=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
  if (var!=NULL && strcmp(n->children->token,var)==0){
    dbg("\n\t[nablaJobParse] primaryExpression hits returned argument");
    nprintf(nabla, NULL, "%s_per_thread[tid]",var);
  }
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

  // On inclue les fichiers knAvx.h||knMic.h
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
