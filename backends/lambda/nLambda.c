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
// * CALLS
// ****************************************************************************
const static nWhatWith nLambdaHeaderDefines[]={
  {"NABLA_NB_GLOBAL","1"},
  {"Bool", "bool"},
  {"Integer", "int"},
  {"real", "Real"},
  {"rabs(a)","fabs(a)"},
  {"set(a)", "a"},
  {"set1(cst)", "cst"},
  {"square_root(u)", "sqrt(u)"},
  {"cube_root(u)", "cbrt(u)"},
  {"store(u,_u)", "(*u=_u)"},
  {"load(u)", "(*u)"},
  {"zero()", "0.0"},
  {"norm(u)","::fabs(u)"},
  {"DBG_MODE", "(false)"},
  {"DBG_LVL", "(DBG_INI)"},
  {"DBG_OFF", "0x0000ul"},
  {"DBG_CELL_VOLUME", "0x0001ul"},
  {"DBG_CELL_CQS", "0x0002ul"},
  {"DBG_GTH", "0x0004ul"},
  {"DBG_NODE_FORCE", "0x0008ul"},
  {"DBG_INI_EOS", "0x0010ul"},
  {"DBG_EOS", "0x0020ul"},
  {"DBG_DENSITY", "0x0040ul"},
  {"DBG_MOVE_NODE", "0x0080ul"},
  {"DBG_INI", "0x0100ul"},
  {"DBG_INI_CELL", "0x0200ul"},
  {"DBG_INI_NODE", "0x0400ul"},
  {"DBG_LOOP", "0x0800ul"},
  {"DBG_FUNC_IN", "0x1000ul"},
  {"DBG_FUNC_OUT", "0x2000ul"},
  {"DBG_VELOCITY", "0x4000ul"},
  {"DBG_BOUNDARIES", "0x8000ul"},
  {"DBG_ALL", "0xFFFFul"},
  {"opAdd(u,v)", "(u+v)"},
  {"opSub(u,v)", "(u-v)"},
  {"opDiv(u,v)", "(u/v)"},
  {"opMul(u,v)", "(u*v)"},
  {"opMod(u,v)", "(u%v)"},
  {"opScaMul(u,v)","dot3(u,v)"},
  {"opVecMul(u,v)","cross(u,v)"},    
  {"dot", "dot3"},
  {"ReduceMinToDouble(a)","a"},
  {"ReduceMaxToDouble(a)","a"},
  {"knAt(a)",""},
  {"fatal(a,b)","exit(-1)"},
  //{"synchronize(a)",""},//"_Pragma(\"omp barrier\")"},
  {"mpi_reduce(how,what)","how##ToDouble(what)"},
  //{"reduce(how,what)","how##ToDouble(what)"},
  {"xyz","int"},
  {"GlobalIteration", "global_iteration[0]"},
  {"MD_DirX","0"},
  {"MD_DirY","1"},
  {"MD_DirZ","2"},
  {"File", "std::ofstream&"},
  {"file(name,ext)", "std::ofstream name(#name \".\" #ext)"},
  {"xs_node_cell(c)", "node_cell[n*NABLA_NODE_PER_CELL+c]"},
  {NULL,NULL}
};

const char* nLambdaHeaderForwards[]={NULL};

const nWhatWith nLambdaHeaderTypedef[]={
  {"int","integer"},
  {"double","real"},
  {"struct real3","Real3"},
  {NULL,NULL}
};

const nFwdDefTypes nLambdaHeader={
  nLambdaHeaderForwards,
  nLambdaHeaderDefines,
  nLambdaHeaderTypedef
};

const static nCallSimd lambdaSimdCalls={
  lambdaHookBits,
  lambdaHookGather,
  lambdaHookScatter,
  lambdaHookIncludes
};

const static nCallParallel lambdaCilkCalls={
  nLambdaParallelCilkSync,
  nLambdaParallelCilkSpawn,
  nLambdaParallelCilkLoop,
  nLambdaParallelCilkIncludes
};

const static nCallParallel lambdaOpenMPCalls={
  nLambdaParallelOpenMPSync,
  nLambdaParallelOpenMPSpawn,
  nLambdaParallelOpenMPLoop,
  nLambdaParallelOpenMPIncludes
};

const static nCallParallel lambdaVoidCalls={
  nLambdaParallelVoidSync,
  nLambdaParallelVoidSpawn,
  nLambdaParallelVoidLoop,
  nLambdaParallelVoidIncludes
};

nCalls nLambdaCalls={
  &nLambdaHeader,
  &lambdaSimdCalls,
  &lambdaVoidCalls,
};


// ****************************************************************************
// * HOOKS
// ****************************************************************************
const static nHookXyz lambdaXyzHooks={
  lambdaHookPrevCell,
  lambdaHookNextCell
};

const static nHookPragma lambdaPragmaICCHooks ={
  lambdaHookPragmaIccAlign
};

const static nHookPragma lambdaPragmaGCCHooks={
  lambdaHookPragmaGccAlign
};

// Hooks pour le header
const static nHookHeader nLHookHeader={
  nLambdaHookHeaderDump,
  nLambdaHookHeaderOpen,
  nLambdaHookHeaderDefineEnumerates,
  nLambdaHookHeaderPrefix,
  nLambdaHookHeaderIncludes,
  nLambdaHookHeaderPostfix
};

// Hooks pour le source
const static nHookSource nLHookSource={
  nLambdaHookSourceOpen,
  nLambdaHookSourceInclude
};
  
// Hooks pour le maillage
const static nHookMesh nLHookMesh={
  nLambdaHookMeshPrefix,
  nLambdaHookMeshCore,
  nLambdaHookMeshPostfix
};
  
// Hooks pour les variables
const static nHookVars nLHookVars={
  nLambdaHookVarsInit,
  nLambdaHookVariablesPrefix,
  nLambdaHookVariablesPostfix
};  

// Hooks pour le main
const static nHookMain nLHookMain={
  nLambdaHookMainPrefix,
  nLambdaHookMainPreInit,
  nLambdaHookMainVarInitKernel,
  nLambdaHookMainVarInitCall,
  nLambdaHookMain,
  nLambdaHookMainPostInit,
  nLambdaHookMainPostfix
};  

const static nHookForAll nLHookForAll={
  lambdaHookPrefixEnumerate,
  lambdaHookDumpEnumerate,
  lambdaHookItem,
  lambdaHookPostfixEnumerate
};

const static nHookToken nLHookToken={
  lambdaHookSwitchToken,
  lambdaHookTurnTokenToVariable,
  lambdaHookTurnTokenToOption,
  lambdaHookSystem,
  lambdaHookIteration,
  lambdaHookExit,
  lambdaHookTime,
  lambdaHookFatal,
  lambdaHookTurnBracketsToParentheses
};

const static nHookGrammar hookGrammar={
  lambdaHookFunction,
  lambdaHookJob,
  lambdaHookReduction,
  lambdaHookPrimaryExpressionToReturn,
  lambdaHookReturnFromArgument
};

const static nHookCall nLambdaHookCall={
  lambdaHookAddCallNames,
  lambdaHookAddArguments,
  lambdaHookEntryPointPrefix,
  lambdaHookDfsForCalls,
  lambdaHookAddExtraParameters,
  lambdaHookDumpNablaParameterList
};

static nHooks nLambdaHooks={
  &nLHookForAll,
  &nLHookToken,
  &hookGrammar,
  &nLambdaHookCall,
  &lambdaXyzHooks,
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
    nLambdaCalls.parallel=&lambdaCilkCalls;
  
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nLambdaCalls.parallel=&lambdaOpenMPCalls;
  
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nLambdaHooks.pragma=&lambdaPragmaICCHooks;

  nabla->call=&nLambdaCalls;
  return &nLambdaHooks;
  
}
