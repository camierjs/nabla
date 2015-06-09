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
// * nOkina
// ****************************************************************************
NABLA_STATUS nOkina(nablaMain *nabla,
                      astNode *root,
                      const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  // Définition des hooks pour le mode Standard
  nablaBackendSimdHooks nablaOkinaSimdStdHooks={
    nOkinaStdBits,
    nOkinaStdGather,
    nOkinaStdScatter,
    nOkinaStdTypedef,
    nOkinaStdDefines,
    nOkinaStdForwards,
    nOkinaStdPrevCell,
    nOkinaStdNextCell,
    nOkinaStdIncludes
  };
  // Définition des hooks pour le mode SSE
  nablaBackendSimdHooks nablaOkinaSimdSseHooks={
    nOkinaSseBits,
    nOkinaSseGather,
    nOkinaSseScatter,
    nOkinaSseTypedef,
    nOkinaSseDefines,
    nOkinaSseForwards,
    nOkinaSsePrevCell,
    nOkinaSseNextCell,
    nOkinaSseIncludes
  };
  // Définition des hooks pour le mode AVX
  nablaBackendSimdHooks nablaOkinaSimdAvxHooks={
    nOkinaAvxBits,
    nOkinaAvxGather,
    nOkinaAvxScatter,
    nOkinaAvxTypedef,
    nOkinaAvxDefines,
    nOkinaAvxForwards,
    nOkinaAvxPrevCell,
    nOkinaAvxNextCell,
    nOkinaAvxIncludes
  };
  // Définition des hooks pour le mode MIC
  nablaBackendSimdHooks nablaOkinaSimdMicHooks={ 
    nOkinaMicBits,
    nOkinaMicGather,
    nOkinaMicScatter,
    nOkinaMicTypedef,
    nOkinaMicDefines,
    nOkinaMicForwards,
    nOkinaMicPrevCell,
    nOkinaMicNextCell,
    nOkinaMicIncludes
  };
  // Définition des hooks pour Cilk+
  nablaBackendParallelHooks okinaCilkHooks={
    nOkinaParallelCilkSync,
    nOkinaParallelCilkSpawn,
    nOkinaParallelCilkLoop,
    nOkinaParallelCilkIncludes
  };
  // Définition des hooks pour OpenMP
  nablaBackendParallelHooks okinaOpenMPHooks={
    nOkinaParallelOpenMPSync,
    nOkinaParallelOpenMPSpawn,
    nOkinaParallelOpenMPLoop,
    nOkinaParallelOpenMPIncludes
  };
  // Définition des hooks quand il n'y a pas de parallélisation
  nablaBackendParallelHooks okinaVoidHooks={
    nOkinaParallelVoidSync,
    nOkinaParallelVoidSpawn,
    nOkinaParallelVoidLoop,
    nOkinaParallelVoidIncludes
  };
  // Pragmas hooks definition for ICC or GCC
  nablaBackendPragmaHooks okinaPragmaICCHooks ={
    nOkinaPragmaIccIvdep,
    nOkinaPragmaIccAlign
  };
  nablaBackendPragmaHooks okinaPragmaGCCHooks={
    nOkinaPragmaGccIvdep,
    nOkinaPragmaGccAlign
  };
  // Definition of Okina's Hooks
  nablaBackendHooks okinaBackendHooks={
    nOkinaHookEnumeratePrefix,
    nOkinaHookEnumerateDumpXYZ,
    nOkinaHookEnumerateDump,
    nOkinaHookEnumeratePostfix,
    nOkinaHookItem,
    nOkinaHookTokenSwitch,
    nOkinaHookVariablesTurnTokenToVariable,
    nOkinaHookVariablesSystem,
    nOkinaHookParamsAddExtra,
    nOkinaHookParamsDumpList,
    nOkinaHookVariablesTurnBracketsToParentheses,
    nOkinaHookDiffraction,
    nOkinaHookFunctionName,
    nOkinaHookFunction,
    nOkinaHookJob,
    nOkinaHookReduction,
    nOkinaHookIteration,
    nOkinaHookExit,
    nOkinaHookTime,
    nOkinaHookFatal,
    nOkinaHookAddCallNames,
    nOkinaHookAddArguments,
    nOkinaHookTurnTokenToOption,
    nOkinaHookEntryPointPrefix,
    nOkinaHookDfsForCalls,
    nOkinaHookPrimaryExpressionToReturn,
    nOkinaHookReturnFromArgument
  };
  // Switch between STD, SSE, AVX, MIC
  // Par défaut, on est en mode 'std'
  nabla->simd=&nablaOkinaSimdStdHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
    nabla->simd=&nablaOkinaSimdSseHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
    nabla->simd=&nablaOkinaSimdAvxHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
    nabla->simd=&nablaOkinaSimdAvxHooks;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
    nabla->simd=&nablaOkinaSimdMicHooks;

  // Switch between parallel modes
  // By default, we have no parallelization
  nabla->parallel=&okinaVoidHooks;
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nabla->parallel=&okinaCilkHooks;
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nabla->parallel=&okinaOpenMPHooks;

  // Switch between ICC or GCC pragmas
  // Par defaut, on met GCC
  nabla->pragma=&okinaPragmaGCCHooks;
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nabla->pragma=&okinaPragmaICCHooks;

  // Set the hooks for this backend
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
  
  // Dump dans le HEADER: includes, typedefs, defines, debug, maths & errors stuff
  nOkinaHeaderPrefix(nabla);
  nOkinaHeaderIncludes(nabla);
  nablaDefines(nabla,nabla->simd->defines);
  nablaTypedefs(nabla,nabla->simd->typedefs);
  nablaForwards(nabla,nabla->simd->forwards);

  // On inclue les fichiers kn'SIMD'
  nOkinaHeaderSimd(nabla);
  nOkinaHeaderDbg(nabla);
  nOkinaHeaderMth(nabla);

  // Dump dans le fichier SOURCE de l'include de l'entity
  nOkinaHeaderInclude(nabla);

  // Parse du code préprocessé et lance les hooks associés
  nablaMiddlendParseAndHook(root,nabla);
  
  // On rajoute le kernel d'initialisation des variable
  nOkinaInitVariables(nabla);

  // Mesh structures and functions depends on the ℝ library that can be used
  if ((nabla->entity->libraries&(1<<real))!=0){
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
