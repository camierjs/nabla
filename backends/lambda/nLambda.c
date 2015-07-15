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


// ****************************************************************************
// * nLambda
// ****************************************************************************
NABLA_STATUS nLambda(nablaMain *nabla,
                     astNode *root,
                     const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  // Définition des hooks pour l'AVX ou le MIC
  nablaBackendSimdHooks nablaCcSimdStdHooks={
    ccHookBits,
    ccHookGather,
    ccHookScatter,
    ccTypedef,
    ccDefines,
    ccForwards,
    ccHookPrevCell,
    ccHookNextCell,
    ccHookIncludes
  };
  nabla->simd=&nablaCcSimdStdHooks;
    
  // Définition des hooks pour Cilk+ *ou pas*
  nablaBackendParallelHooks ccCilkHooks={
    nLambdaParallelCilkSync,
    nLambdaParallelCilkSpawn,
    nLambdaParallelCilkLoop,
    nLambdaParallelCilkIncludes
  };
  nablaBackendParallelHooks ccOpenMPHooks={
    nLambdaParallelOpenMPSync,
    nLambdaParallelOpenMPSpawn,
    nLambdaParallelOpenMPLoop,
    nLambdaParallelOpenMPIncludes
  };
  nablaBackendParallelHooks ccVoidHooks={
    nLambdaParallelVoidSync,
    nLambdaParallelVoidSpawn,
    nLambdaParallelVoidLoop,
    nLambdaParallelVoidIncludes
  };
  nabla->parallel=&ccVoidHooks;
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nabla->parallel=&ccCilkHooks;
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nabla->parallel=&ccOpenMPHooks;

  
  nablaBackendPragmaHooks ccPragmaICCHooks ={
    ccHookPragmaIccIvdep,
    ccHookPragmaIccAlign
  };
  nablaBackendPragmaHooks ccPragmaGCCHooks={
    ccHookPragmaGccIvdep,
    ccHookPragmaGccAlign
  };
  // Par defaut, on met GCC
  nabla->pragma=&ccPragmaGCCHooks;
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nabla->pragma=&ccPragmaICCHooks;
  
  static nablaBackendHooks ccBackendHooks={
    // Jobs stuff
    ccHookPrefixEnumerate,
    ccHookDumpEnumerateXYZ,
    ccHookDumpEnumerate,
    ccHookPostfixEnumerate,
    ccHookItem,
    ccHookSwitchToken,
    ccHookTurnTokenToVariable,
    ccHookSystem,
    ccHookAddExtraParameters,
    ccHookDumpNablaParameterList,
    ccHookTurnBracketsToParentheses,
    ccHookJobDiffractStatement,
    // Other hooks
    ccHookFunctionName,
    ccHookFunction,
    ccHookJob,
    ccHookReduction,
    ccHookIteration,
    ccHookExit,
    ccHookTime,
    ccHookFatal,
    ccHookAddCallNames,
    ccHookAddArguments,
    ccHookTurnTokenToOption,
    ccHookEntryPointPrefix,
    ccHookDfsForCalls,
    ccHookPrimaryExpressionToReturn,
    ccHookReturnFromArgument
  };
  nabla->hook=&ccBackendHooks;

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
  
  // Dump dans le HEADER des includes, typedefs, defines, debug, maths & errors stuff
  ccHeaderPrefix(nabla);
  ccHeaderIncludes(nabla);
  nMiddleDefines(nabla,nabla->simd->defines);
  nMiddleTypedefs(nabla,nabla->simd->typedefs);
  nMiddleForwards(nabla,nabla->simd->forwards);

  // On inclue les fichiers kn'SIMD'
  ccHeaderSimd(nabla);
  ccHeaderDbg(nabla);
  ccHeaderMth(nabla);
  ccMesh(nabla);
  ccDefineEnumerates(nabla);

  // Dump dans le fichier SOURCE
  ccInclude(nabla);
  
  // Parse du code préprocessé et lance les hooks associés
  nMiddleParseAndHook(root,nabla);
  ccMainVarInitKernel(nabla);

  // Partie PREFIX
  ccMainPrefix(nabla);
  ccVariablesPrefix(nabla);
  ccMainMeshPrefix(nabla);
  
  // Partie Pré Init
  ccMainPreInit(nabla);
  ccMainVarInitCall(nabla);
      
  // Dump des entry points dans le main
  ccMain(nabla);

  // Partie Post Init
  ccMainPostInit(nabla);
  
  // Partie POSTFIX
  ccHeaderPostfix(nabla); 
  ccMainMeshPostfix(nabla);
  ccVariablesPostfix(nabla);
  ccMainPostfix(nabla);
  return NABLA_OK;
}

