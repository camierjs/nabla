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
#include "backends/lambda/lambda.h"
#include "backends/x86/hook/hook.h"


const static callParallel lambdaCilkCalls={
  nLambdaParallelCilkSync,
  nLambdaParallelCilkSpawn,
  nLambdaParallelCilkLoop,
  nLambdaParallelCilkIncludes
};

const static callParallel lambdaOpenMPCalls={
  nLambdaParallelOpenMPSync,
  nLambdaParallelOpenMPSpawn,
  nLambdaParallelOpenMPLoop,
  nLambdaParallelOpenMPIncludes
};

const static callParallel lambdaVoidCalls={
  nLambdaParallelVoidSync,
  nLambdaParallelVoidSpawn,
  nLambdaParallelVoidLoop,
  nLambdaParallelVoidIncludes
};

backendCalls lambdaCalls={
  NULL,
  NULL,
  &lambdaVoidCalls,
};


// ****************************************************************************
// * HOOKS
// ****************************************************************************
const static hookForAll forall={
  NULL,
  lambdaHookForAllDump,
  lambdaHookForAllItem,
  lambdaHookForAllPostfix
};

const static hookToken token={
  NULL,
  lambdaHookSwitchToken,
  lambdaHookTurnTokenToVariable,
  lambdaHookTurnTokenToOption,
  lambdaHookSystem,
  lambdaHookIteration,
  xHookExit,
  xHookTime,
  xHookFatal,
  lambdaHookTurnBracketsToParentheses,
  lambdaHookIsTest,
  NULL
};

const static hookGrammar gram={
  NULL,
  NULL,
  lambdaHookReduction,
  NULL,
  NULL,
  xHookDfsVariable
};

const static hookCall call={
  xHookAddCallNames,
  xHookAddArguments,
  xHookEntryPointPrefix,
  xHookDfsForCalls,
  NULL,
  NULL
};







const static hookXyz xyz={
  NULL,
  xHookPrevCell,
  xHookNextCell,
  xHookSysPostfix
};

// Hooks pour le header
const static hookHeader header={
  nLambdaHookHeaderDump,
  nLambdaHookHeaderOpen,
  nLambdaHookHeaderDefineEnumerates,
  nLambdaHookHeaderPrefix,
  nLambdaHookHeaderIncludes,
  nLambdaHookHeaderPostfix
};

// Hooks pour le source
const static hookSource source={
  lHookSourceOpen,
  lHookSourceInclude,
  lHookSourceNamespace
};
  
// Hooks pour le maillage
const static hookMesh mesh={
  nLambdaHookMeshPrefix,
  nLambdaHookMeshCore,
  nLambdaHookMeshPostfix
};
  
// Hooks pour les variables
const static hookVars vars={
  nLambdaHookVariablesInit,
  nLambdaHookVariablesPrefix,
  nLambdaHookVariablesMalloc,
  nLambdaHookVariablesFree
};  

// Hooks pour le main
const static hookMain mains={
  nLambdaHookMainPrefix,
  nLambdaHookMainPreInit,
  nLambdaHookMainVarInitKernel,
  nLambdaHookMainVarInitCall,
  nLambdaHookMain,
  nLambdaHookMainPostInit,
  nLambdaHookMainPostfix
};  

static hooks lambdaHooks={
  &forall,
  &token,
  &gram,
  &call,
  &xyz,
  NULL,
  &header,
  &source,
  &mesh,
  &vars,
  &mains
};


// ****************************************************************************
// * nLambda
// ****************************************************************************
hooks* lambda(nablaMain *nabla){
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    lambdaCalls.parallel=&lambdaCilkCalls;
  
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    lambdaCalls.parallel=&lambdaOpenMPCalls;
  
  nabla->call=&lambdaCalls;
  return &lambdaHooks;
  
}
