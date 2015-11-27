///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 201~2015 CEA/DAM/DIF                                       //
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


// ****************************************************************************
// * nOkina
// ****************************************************************************
NABLA_STATUS nOkina(nablaMain *nabla,
                      astNode *root,
                      const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  const callHeader nablaOkinaHeaderStdHeader={
    nOkinaStdForwards,
    nOkinaStdDefines,
    nOkinaStdTypedef
  };
  const callHeader nablaOkinaHeaderSseHeader={
    nOkinaSseForwards,
    nOkinaSseDefines,
    nOkinaSseTypedef
  };
  const callHeader nablaOkinaHeaderAvxHeader={
    nOkinaAvxForwards,
    nOkinaAvxDefines,
    nOkinaAvxTypedef
  };
  const callHeader nablaOkinaHeaderMicHeader={
    nOkinaMicForwards,
    nOkinaMicDefines,
    nOkinaMicTypedef
  };
  // Std Typedefs, Defines & Forwards
  const hookHeader nablaOkinaHeaderStdHooks={
    NULL, // dump
    NULL, // open
    NULL, // enums
    NULL, // prefix
    NULL, // include
    NULL  // postfix
  };
  // Sse Typedefs, Defines & Forwards
  const hookHeader nablaOkinaHeaderSseHooks={
    NULL, // dump
    NULL, // open
    NULL, // enums
    NULL, // prefix
    NULL, // include
    NULL  // postfix
  };
  // Avx Typedefs, Defines & Forwards
  const hookHeader nablaOkinaHeaderAvxHooks={
    NULL, // dump
    NULL, // open
    NULL, // enums
    NULL, // prefix
    NULL, // include
    NULL  // postfix
  };
  // Mic Typedefs, Defines & Forwards
  const hookHeader nablaOkinaHeaderMicHooks={
    NULL, // dump
    NULL, // open
    NULL, // enums
    NULL, // prefix
    NULL, // include
    NULL  // postfix
  };
  // Définition des hooks pour le mode Standard
  const callSimd nablaOkinaSimdStdCalls={
    nOkinaStdBits,
    nOkinaStdGather,
    nOkinaStdScatter,
    nOkinaStdIncludes
  };
  // Définition des calls pour le mode SSE
  const callSimd nablaOkinaSimdSseCalls={
    nOkinaSseBits,
    nOkinaSseGather,
    nOkinaSseScatter,
    nOkinaSseIncludes
  };
  // Définition des calls pour le mode AVX
  const callSimd nablaOkinaSimdAvxCalls={
    nOkinaAvxBits,
    nOkinaAvxGather,
    nOkinaAvxScatter,
    nOkinaAvxIncludes
  };
  // Définition des calls pour le mode MIC
  const callSimd nablaOkinaSimdMicCalls={ 
    nOkinaMicBits,
    nOkinaMicGather,
    nOkinaMicScatter,
    nOkinaMicIncludes
  };
  // Définition des hooks des directions
  const hookXyz nablaOkinaXyzStdHooks={
    nOkinaHookSysPrefix,
    nOkinaStdPrevCell,
    nOkinaStdNextCell,
    nOkinaHookSysPostfix
  };
  // Définition des hooks des directions
  const hookXyz nablaOkinaXyzSseHooks={
    nOkinaHookSysPrefix,
    nOkinaSsePrevCell,
    nOkinaSseNextCell,
    nOkinaHookSysPostfix
  };
  // Définition des hooks pour le mode AVX
  const hookXyz nablaOkinaXyzAvxHooks={
    nOkinaHookSysPrefix,
    nOkinaAvxPrevCell,
    nOkinaAvxNextCell,
    nOkinaHookSysPostfix
  };
  // Définition des hooks pour le mode MIC
  const hookXyz nablaOkinaXyzMicHooks={ 
    nOkinaHookSysPrefix,
    nOkinaMicPrevCell,
    nOkinaMicNextCell,
    nOkinaHookSysPostfix
  };
  // Définition des calls pour Cilk+
  const callParallel okinaCilkCalls={
    nOkinaParallelCilkSync,
    nOkinaParallelCilkSpawn,
    nOkinaParallelCilkLoop,
    nOkinaParallelCilkIncludes
  };
  // Définition des calls pour OpenMP
  const callParallel okinaOpenMPCalls={
    nOkinaParallelOpenMPSync,
    nOkinaParallelOpenMPSpawn,
    nOkinaParallelOpenMPLoop,
    nOkinaParallelOpenMPIncludes
  };
  // Définition des calls quand il n'y a pas de parallélisation
  const callParallel okinaVoidCalls={
    nOkinaParallelVoidSync,
    nOkinaParallelVoidSpawn,
    nOkinaParallelVoidLoop,
    nOkinaParallelVoidIncludes
  };
  // Pragmas hooks definition for ICC or GCC
  const hookPragma okinaPragmaICCHooks ={
    nOkinaPragmaIccAlign
  };
  const hookPragma okinaPragmaGCCHooks={
    nOkinaPragmaGccAlign
  };
  const hookForAll nOkinaHookForAll={
    nOkinaHookEnumeratePrefix,
    nOkinaHookEnumerateDump,
    nOkinaHookItem,
    nOkinaHookEnumeratePostfix
  };
  const hookToken nOkinaHookToken={
    nOkinaHookTokenPrefix,
    nOkinaHookTokenSwitch,
    nOkinaHookVariablesTurnTokenToVariable,
    nOkinaHookTurnTokenToOption,
    nOkinaHookVariablesSystem,
    nOkinaHookIteration,
    nOkinaHookExit,
    nOkinaHookTime,
    nOkinaHookFatal,
    nOkinaHookVariablesTurnBracketsToParentheses,
    okinaHookIsTest,
    nOkinaHookTokenPostfix
  };
  const hookGrammar nOkinaHookGrammar={
    nOkinaHookFunction,
    nOkinaHookJob,
    nOkinaHookReduction,
    nOkinaHookPrimaryExpressionToReturn,
    nOkinaHookReturnFromArgument
  };
  const hookCall nOkinaHookCall={
    nOkinaHookAddCallNames,
    nOkinaHookAddArguments,
    nOkinaHookEntryPointPrefix,
    nOkinaHookDfsForCalls,
    nOkinaHookParamsAddExtra,
    nOkinaHookParamsDumpList
  };
  // Definition of Okina's Hooks
  hooks okinaBackendHooks={
    &nOkinaHookForAll,
    &nOkinaHookToken,
    &nOkinaHookGrammar,
    &nOkinaHookCall,
    NULL, // xyz
    NULL, // pragma
    NULL, // header
    NULL, // source
    NULL, // mesh
    NULL, // vars
    NULL // main
  };
  // Par défaut, on est en mode 'std'
  calls okinaBackendCalls={
    &nablaOkinaHeaderStdHeader,
    &nablaOkinaSimdStdCalls,
    &okinaVoidCalls // parallel
  };
  
  nabla->call=&okinaBackendCalls;
  nabla->hook=&okinaBackendHooks;

  // Switch between STD, SSE, AVX, MIC
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE){
    nabla->call->simd=&nablaOkinaSimdSseCalls;
    nabla->call->header=&nablaOkinaHeaderSseHeader;
  }
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX){
    nabla->call->simd=&nablaOkinaSimdAvxCalls;
    nabla->call->header=&nablaOkinaHeaderAvxHeader;
  }
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2){
    nabla->call->simd=&nablaOkinaSimdAvxCalls;
    nabla->call->header=&nablaOkinaHeaderAvxHeader;
  }
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC){
    nabla->call->simd=&nablaOkinaSimdMicCalls;
    nabla->call->header=&nablaOkinaHeaderMicHeader;
  }

  // Gestion des directions
  nabla->hook->xyz=&nablaOkinaXyzStdHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
    nabla->hook->xyz=&nablaOkinaXyzSseHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
    nabla->hook->xyz=&nablaOkinaXyzAvxHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
    nabla->hook->xyz=&nablaOkinaXyzAvxHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
    nabla->hook->xyz=&nablaOkinaXyzMicHooks;

  // Gestion du header
  nabla->hook->header=&nablaOkinaHeaderStdHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
    nabla->hook->header=&nablaOkinaHeaderSseHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
    nabla->hook->header=&nablaOkinaHeaderAvxHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
    nabla->hook->header=&nablaOkinaHeaderAvxHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
    nabla->hook->header=&nablaOkinaHeaderMicHooks;

  // Switch between parallel modes
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nabla->call->parallel=&okinaCilkCalls;
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nabla->call->parallel=&okinaOpenMPCalls;

  // Switch between ICC or GCC pragmas
  // Par defaut, on met GCC
  nabla->hook->pragma=&okinaPragmaGCCHooks;
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nabla->hook->pragma=&okinaPragmaICCHooks;


  // Rajout de la variable globale 'iteration'
  {
    nablaVariable *iteration = nMiddleVariableNew(nabla);
    nMiddleVariableAdd(nabla, iteration);
    iteration->axl_it=false;
    iteration->item=strdup("global");
    iteration->type=strdup("integer");
    iteration->name=strdup("iteration");
  }
 
  // Ouverture du fichier source du entity
  sprintf(srcFileName, "%s.cc", nabla->name);
  if ((nabla->entity->src=fopen(srcFileName, "w")) == NULL) exit(NABLA_ERROR);

  // Ouverture du fichier header du entity
  sprintf(hdrFileName, "%s.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
  
  // Dump dans le HEADER: includes, typedefs, defines, debug, maths & errors stuff
  nOkinaHeaderPrefix(nabla);
  nOkinaHeaderIncludes(nabla);
  nMiddleDefines(nabla,nabla->call->header->defines);
  nMiddleTypedefs(nabla,nabla->call->header->typedefs);
  nMiddleForwards(nabla,nabla->call->header->forwards);

  // On inclue les fichiers kn'SIMD'
  nOkinaHeaderSimd(nabla);
  nOkinaHeaderDbg(nabla);
  nOkinaHeaderMth(nabla);

  // Dump dans le fichier SOURCE de l'include de l'entity
  nOkinaHeaderInclude(nabla);

  // Parse du code préprocessé et lance les hooks associés
  nMiddleGrammar(root,nabla);
  
  // On rajoute le kernel d'initialisation des variable
  nOkinaInitVariables(nabla);

  // Mesh structures and functions depends on the ℝ library that can be used
  if (isWithLibrary(nabla,with_real)){
    dbg("\n\t[nOkina] okinaMesh 1D !");
    nOkinaMesh1D(nabla);
  }else{
    nOkinaMesh3D(nabla);
    dbg("\n\t[nOkina] okinaMesh 3D !");
  }
  nOkinaEnumDefine(nabla);
  
  // Partie PREFIX
  nOkinaMainPrefix(nabla);
  okinaVariablesPrefix(nabla);
  nOkinaMeshPrefix(nabla);

  // Partie Pré Init
  nOkinaMainPreInit(nabla);
  nOkinaInitVariableDbg(nabla);
      
  // Dump des entry points dans le main
  nOkinaMain(nabla);

  // Partie Post Init
  nOkinaMainPostInit(nabla);
  
  // Partie POSTFIX
  nOkinaHeaderPostfix(nabla); 
  nOkinaMeshPostfix(nabla);
  okinaVariablesPostfix(nabla);
  nOkinaMainPostfix(nabla);
  return NABLA_OK;
}
