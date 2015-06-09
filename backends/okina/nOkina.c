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
#include "frontend/nablaAst.h"



// ****************************************************************************
// * okinaPrimaryExpressionToReturn
// ****************************************************************************
static bool okinaHookPrimaryExpressionToReturn(nablaMain *nabla, nablaJob *job, astNode *n){
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
static void okinaHookReturnFromArgument(nablaMain *nabla, nablaJob *job){
  const char *rtnVariable=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
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
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nabla->parallel=&okinaCilkHooks;
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
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
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
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
    okinaHookIteration,
    okinaHookExit,
    okinaHookTime,
    okinaHookFatal,
    okinaHookAddCallNames,
    okinaHookAddArguments,
    okinaHookTurnTokenToOption,
    okinaHookEntryPointPrefix,
    okinaHookDfsForCalls,
    okinaHookPrimaryExpressionToReturn,
    okinaHookReturnFromArgument
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

  // Dump dans le fichier SOURCE
  okinaInclude(nabla);

  // Parse du code préprocessé et lance les hooks associés
  nablaMiddlendParseAndHook(root,nabla);
  
  // On rajoute le kernel d'initialisation des variable
  nccOkinaMainVarInitKernel(nabla);

  // Mesh structures and functions depends on the ℝ library that can be used
  if ((nabla->entity->libraries&(1<<real))!=0){
    dbg("\n\t[nccOkina] okinaMesh 1D !");
    okinaMesh1D(nabla);
  }else{
    okinaMesh3D(nabla);
    dbg("\n\t[nccOkina] okinaMesh 3D !");
  }
  okinaDefineEnumerates(nabla);
  
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
