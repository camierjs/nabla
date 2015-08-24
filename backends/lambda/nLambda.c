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

const nHookSimd lambdaSimdHooks={
  lambdaHookBits,
  lambdaHookGather,
  lambdaHookScatter,
  lambdaHookPrevCell, // nMiddleVariables
  lambdaHookNextCell, // nMiddleVariables
  lambdaHookIncludes
};
  
const nHookParallel lambdaCilkHooks={
  nLambdaParallelCilkSync,
  nLambdaParallelCilkSpawn,
  nLambdaParallelCilkLoop,
  nLambdaParallelCilkIncludes
};

const nHookParallel lambdaOpenMPHooks={
  nLambdaParallelOpenMPSync,
  nLambdaParallelOpenMPSpawn,
  nLambdaParallelOpenMPLoop,
  nLambdaParallelOpenMPIncludes
};

const nHookParallel lambdaVoidHooks={
  nLambdaParallelVoidSync,
  nLambdaParallelVoidSpawn,
  nLambdaParallelVoidLoop,
  nLambdaParallelVoidIncludes
};

const nHookPragma lambdaPragmaICCHooks ={
  lambdaHookPragmaIccIvdep,
  lambdaHookPragmaIccAlign // nMiddleFunctions
};

const nHookPragma lambdaPragmaGCCHooks={
  lambdaHookPragmaGccIvdep,
  lambdaHookPragmaGccAlign // nMiddleFunctions
};

// Hooks pour le header
const nHookHeader nLHookHeader={
  nLambdaHookForwards,
  nLambdaHookDefines,
  nLambdaHookTypedef,
  nLambdaHookHeaderDump, // nMiddleAnimate
  nLambdaHookHeaderOpen, // nMiddleAnimate
  nLambdaHookHeaderDefineEnumerates, // nMiddleAnimate
  nLambdaHookHeaderPrefix, // nMiddleAnimate
  nLambdaHookHeaderIncludes, // nMiddleAnimate
  nLambdaHookHeaderPostfix // nMiddleAnimate
};

// Hooks pour le source
const nHookSource nLHookSource={
  nLambdaHookSourceOpen, // nMiddleAnimate
  nLambdaHookSourceInclude // nMiddleAnimate
};
  
// Hooks pour le maillage
const nHookMesh nLHookMesh={
  nLambdaHookMeshPrefix, // nMiddleAnimate
  nLambdaHookMeshCore, // nMiddleAnimate
  nLambdaHookMeshPostfix // nMiddleAnimate
};
  
// Hooks pour les variables
const nHookVars nLHookVars={
  nLambdaHookVarsInit, // nMiddleAnimate
  nLambdaHookVariablesPrefix, // nMiddleAnimate
  nLambdaHookVariablesPostfix // nMiddleAnimate
};  

// Hooks pour le main
const nHookMain nLHookMain={
  nLambdaHookMainPrefix, // nMiddleAnimate
  nLambdaHookMainPreInit, // nMiddleAnimate
  nLambdaHookMainVarInitKernel, // nMiddleAnimate
  nLambdaHookMainVarInitCall, // nMiddleAnimate
  nLambdaHookMain, // nMiddleAnimate
  nLambdaHookMainPostInit, // nMiddleAnimate
  nLambdaHookMainPostfix // nMiddleAnimate
};  

const nHookForAll nLHookForAll={
  lambdaHookPrefixEnumerate, // nMiddleJobs, nMiddleFunctions?
  lambdaHookDumpEnumerate, // nMiddleJobs, nMiddleFunctions?
  lambdaHookItem, // nMiddleJobs
  lambdaHookPostfixEnumerate // nMiddleJobs, nMiddleFunctions?
};

const nHookToken nLHookToken={
  lambdaHookSwitchToken, // nMiddleJobs
  lambdaHookTurnTokenToVariable, // nMiddleJobs, nMiddleFunctions
  lambdaHookTurnTokenToOption, // nMiddleOptions
  lambdaHookSystem, // nMiddleVariables
  lambdaHookIteration, // nMiddleFunctions
  lambdaHookExit, // nMiddleFunctions
  lambdaHookTime, // nMiddleFunctions
  lambdaHookFatal, // nMiddleFunctions
  lambdaHookTurnBracketsToParentheses // nMiddleJobs
};

const nHookGrammar hookGrammar={
  lambdaHookFunction, // nMiddleGrammar
  lambdaHookJob, // nMiddleGrammar
  lambdaHookReduction, // nMiddleGrammar
  lambdaHookPrimaryExpressionToReturn, // nMiddleJobs
  lambdaHookReturnFromArgument // nMiddleJobs
};

const nHookCall nLambdaHookCall={
  lambdaHookAddCallNames, // nMiddleFunctions
  lambdaHookAddArguments, // nMiddleFunctions
  lambdaHookEntryPointPrefix, // nMiddleJobs, nMiddleFunctions
  lambdaHookDfsForCalls, // nMiddleFunctions
  lambdaHookAddExtraParameters, // nMiddleJobs, nMiddleFunctions
  lambdaHookDumpNablaParameterList // nMiddleJobs
};

nHooks nLambdaHooks={
  &nLHookForAll,
  &nLHookToken,
  &hookGrammar,
  &nLambdaHookCall,
  &lambdaSimdHooks,
  &lambdaVoidHooks,
  &lambdaPragmaGCCHooks,
  &nLHookHeader,
  &nLHookSource,
  &nLHookMesh,
  &nLHookVars,
  &nLHookMain
};


// ****************************************************************************
// * nLambda
// ****************************************************************************
nHooks *nLambda(nablaMain *nabla){
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nLambdaHooks.parallel=&lambdaCilkHooks;
  
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nLambdaHooks.parallel=&lambdaOpenMPHooks;
  
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nLambdaHooks.pragma=&lambdaPragmaICCHooks;

  return &nLambdaHooks;
}
