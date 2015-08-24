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
#include "frontend/nablaAst.h"


// ****************************************************************************
// * nOkina
// ****************************************************************************
NABLA_STATUS nOkina(nablaMain *nabla,
                      astNode *root,
                      const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  // Std Typedefs, Defines & Forwards
  nHookHeader nablaOkinaHeaderStdHooks={
    nOkinaStdForwards,
    nOkinaStdDefines,
    nOkinaStdTypedef,
    NULL, // dump
    NULL, // open
    NULL, // enums
    NULL, // prefix
    NULL, // include
    NULL  // postfix
  };
  // Sse Typedefs, Defines & Forwards
  nHookHeader nablaOkinaHeaderSseHooks={
    nOkinaSseForwards,
    nOkinaSseDefines,
    nOkinaSseTypedef,
    NULL, // dump
    NULL, // open
    NULL, // enums
    NULL, // prefix
    NULL, // include
    NULL  // postfix
  };
  // Avx Typedefs, Defines & Forwards
  nHookHeader nablaOkinaHeaderAvxHooks={
    nOkinaAvxForwards,
    nOkinaAvxDefines,
    nOkinaAvxTypedef,
    NULL, // dump
    NULL, // open
    NULL, // enums
    NULL, // prefix
    NULL, // include
    NULL  // postfix
  };
  // Mic Typedefs, Defines & Forwards
  nHookHeader nablaOkinaHeaderMicHooks={
    nOkinaMicForwards,
    nOkinaMicDefines,
    nOkinaMicTypedef,
    NULL, // dump
    NULL, // open
    NULL, // enums
    NULL, // prefix
    NULL, // include
    NULL  // postfix
  };
  // Définition des hooks pour le mode Standard
  nHookSimd nablaOkinaSimdStdHooks={
    nOkinaStdBits,
    nOkinaStdGather,
    nOkinaStdScatter,
    nOkinaStdPrevCell,
    nOkinaStdNextCell,
    nOkinaStdIncludes
  };
  // Définition des hooks pour le mode SSE
  nHookSimd nablaOkinaSimdSseHooks={
    nOkinaSseBits,
    nOkinaSseGather,
    nOkinaSseScatter,
    nOkinaSsePrevCell,
    nOkinaSseNextCell,
    nOkinaSseIncludes
  };
  // Définition des hooks pour le mode AVX
  nHookSimd nablaOkinaSimdAvxHooks={
    nOkinaAvxBits,
    nOkinaAvxGather,
    nOkinaAvxScatter,
    nOkinaAvxPrevCell,
    nOkinaAvxNextCell,
    nOkinaAvxIncludes
  };
  // Définition des hooks pour le mode MIC
  nHookSimd nablaOkinaSimdMicHooks={ 
    nOkinaMicBits,
    nOkinaMicGather,
    nOkinaMicScatter,
    nOkinaMicPrevCell,
    nOkinaMicNextCell,
    nOkinaMicIncludes
  };
  // Définition des hooks pour Cilk+
  nHookParallel okinaCilkHooks={
    nOkinaParallelCilkSync,
    nOkinaParallelCilkSpawn,
    nOkinaParallelCilkLoop,
    nOkinaParallelCilkIncludes
  };
  // Définition des hooks pour OpenMP
  nHookParallel okinaOpenMPHooks={
    nOkinaParallelOpenMPSync,
    nOkinaParallelOpenMPSpawn,
    nOkinaParallelOpenMPLoop,
    nOkinaParallelOpenMPIncludes
  };
  // Définition des hooks quand il n'y a pas de parallélisation
  nHookParallel okinaVoidHooks={
    nOkinaParallelVoidSync,
    nOkinaParallelVoidSpawn,
    nOkinaParallelVoidLoop,
    nOkinaParallelVoidIncludes
  };
  // Pragmas hooks definition for ICC or GCC
  nHookPragma okinaPragmaICCHooks ={
    nOkinaPragmaIccIvdep,
    nOkinaPragmaIccAlign
  };
  nHookPragma okinaPragmaGCCHooks={
    nOkinaPragmaGccIvdep,
    nOkinaPragmaGccAlign
  };
  nHookForAll nOkinaHookForAll={
    nOkinaHookEnumeratePrefix,
    nOkinaHookEnumerateDump,
    nOkinaHookItem,
    nOkinaHookEnumeratePostfix
  };
  
  nHookToken nOkinaHookToken={
    nOkinaHookTokenSwitch,
    nOkinaHookVariablesTurnTokenToVariable,
    nOkinaHookTurnTokenToOption,
    nOkinaHookVariablesSystem,
    nOkinaHookIteration,
    nOkinaHookExit,
    nOkinaHookTime,
    nOkinaHookFatal,
    nOkinaHookVariablesTurnBracketsToParentheses
  };

  const nHookGrammar nOkinaHookGrammar={
    nOkinaHookFunction,
    nOkinaHookJob,
    nOkinaHookReduction,
    nOkinaHookPrimaryExpressionToReturn,
    nOkinaHookReturnFromArgument
  };

  const nHookCall nOkinaHookCall={
    nOkinaHookAddCallNames,
    nOkinaHookAddArguments,
    nOkinaHookEntryPointPrefix,
    nOkinaHookDfsForCalls,
    nOkinaHookParamsAddExtra,
    nOkinaHookParamsDumpList
  };
  
  // Definition of Okina's Hooks
  nHooks okinaBackendHooks={
    &nOkinaHookForAll,
    &nOkinaHookToken,
    &nOkinaHookGrammar,
    &nOkinaHookCall,
    NULL, // header
    NULL, // source
    NULL, // mesh
    NULL, // vars
    NULL // main
  };
  
  // Set the hooks for this backend
  nabla->hook=&okinaBackendHooks;

  // Switch between STD, SSE, AVX, MIC
  // Par défaut, on est en mode 'std'
  nabla->hook->simd=&nablaOkinaSimdStdHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
    nabla->hook->simd=&nablaOkinaSimdSseHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
    nabla->hook->simd=&nablaOkinaSimdAvxHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
    nabla->hook->simd=&nablaOkinaSimdAvxHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
    nabla->hook->simd=&nablaOkinaSimdMicHooks;

  
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
  // By default, we have no parallelization
  nabla->hook->parallel=&okinaVoidHooks;
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nabla->hook->parallel=&okinaCilkHooks;
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nabla->hook->parallel=&okinaOpenMPHooks;

  // Switch between ICC or GCC pragmas
  // Par defaut, on met GCC
  nabla->hook->pragma=&okinaPragmaGCCHooks;
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nabla->hook->pragma=&okinaPragmaICCHooks;


  // Rajout de la variable globale 'iteration'
  nablaVariable *iteration = nMiddleVariableNew(nabla);
  nMiddleVariableAdd(nabla, iteration);
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
  
  // Dump dans le HEADER: includes, typedefs, defines, debug, maths & errors stuff
  nOkinaHeaderPrefix(nabla);
  nOkinaHeaderIncludes(nabla);
  nMiddleDefines(nabla,nabla->hook->header->defines);
  nMiddleTypedefs(nabla,nabla->hook->header->typedefs);
  nMiddleForwards(nabla,nabla->hook->header->forwards);

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
