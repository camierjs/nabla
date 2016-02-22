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
#include "backends/kokkos/call/call.h"
#include "backends/kokkos/hook/hook.h"


// ****************************************************************************
// * CALLS
// ****************************************************************************
const static nWhatWith headerDefines[]={
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
  {"zero()", "0.0"},
  {"DBG_MODE", "(false)"},
  {"DBG_LVL", "(DBG_ALL)"},
  {"DBG_OFF", "0x0000ul"},
  {"DBG_INI", "0x0100ul"},
  {"DBG_ALL", "0xFFFFul"},
  {"opAdd(u,v)", "(u+v)"},
  {"opSub(u,v)", "(u-v)"},
  {"opDiv(u,v)", "(u/v)"},
  {"opMul(u,v)", "(u*v)"},
  {"opScaMul(u,v)","dot3(u,v)"},
  {"opVecMul(u,v)","cross(u,v)"},    
  {"ReduceMinToDouble(a)","a"},
  {"GlobalIteration", "global_iteration[0]"},
  {"MD_DirX","0"},
  {"MD_DirY","1"},
  {"MD_DirZ","2"},
  {"MD_Plus","0"},
  {"MD_Negt","4"},
  {"MD_Shift","3"},
  {"MD_Mask","7"},
  {NULL,NULL}
};

const char* headerForwards[]={NULL};

const nWhatWith headerTypedef[]={
  {"int","integer"},
  {"double","real"},
  {"struct real3","Real3"},
  {NULL,NULL}
};

const callHeader headerCalls={
  headerForwards,
  headerDefines,
  headerTypedef
};

const static callSimd simdCalls={
  callBits,
  callGather,
  callScatter,
  callIncludes
};

const static callParallel cilkCalls={
  callParallelCilkSync,
  callParallelCilkSpawn,
  callParallelCilkLoop,
  callParallelCilkIncludes
};

const static callParallel openMPCalls={
  callParallelOpenMPSync,
  callParallelOpenMPSpawn,
  callParallelOpenMPLoop,
  callParallelOpenMPIncludes
};

const static callParallel voidCalls={
  callParallelVoidSync,
  callParallelVoidSpawn,
  callParallelVoidLoop,
  callParallelVoidIncludes
};

backendCalls calls={
  &headerCalls,
  &simdCalls,
  &voidCalls,
};


// ****************************************************************************
// * HOOKS
// ****************************************************************************
const static hookForAll forall={
  hookForAllPrefix,
  hookForAllDump,
  hookForAllItem,
  hookForAllPostfix
};

const static hookToken token={
  hookTokenPrefix,
  hookSwitchToken,
  hookTurnTokenToVariable,
  hookTurnTokenToOption,
  hookSystem,
  hookIteration,
  hookExit,
  hookTime,
  hookFatal,
  hookTurnBracketsToParentheses,
  hookIsTest,
  hookTokenPostfix
};

const static hookGrammar gram={
  hookFunction,
  hookJob,
  hookReduction,
  hookPrimaryExpressionToReturn,
  hookReturnFromArgument,
  hookDfsVariable
};

const static hookCall call={
  hookAddCallNames,
  hookAddArguments,
  hookEntryPointPrefix,
  hookDfsForCalls,
  hookAddExtraParametersDFS,
  hookDumpNablaParameterListDFS
};

const static hookXyz xyz={
  hookSysPrefix,
  hookPrevCell,
  hookNextCell,
  hookSysPostfix
};

const static hookPragma icc ={
  hookPragmaIccAlign
};

const static hookPragma gcc={
  hookPragmaGccAlign
};

// Hooks pour le header
const static hookHeader header={
  hookHeaderDump,
  hookHeaderOpen,
  hookHeaderDefineEnumerates,
  hookHeaderPrefix,
  hookHeaderIncludes,
  hookHeaderPostfix
};

// Hooks pour le source
const static hookSource source={
  hookSourceOpen,
  hookSourceInclude
};
  
// Hooks pour le maillage
const static hookMesh mesh={
  hookMeshPrefix,
  hookMeshCore,
  hookMeshPostfix
};
  
// Hooks pour les variables
const static hookVars vars={
  hookVariablesInit,
  hookVariablesPrefix,
  hookVariablesMalloc,
  hookVariablesFree
};  

// Hooks pour le main
const static hookMain mains={
  hookMainPrefix,
  hookMainPreInit,
  hookMainVarInitKernel,
  hookMainVarInitCall,
  hookMainHLT,
  hookMainPostInit,
  hookMainPostfix
};  

static backendHooks hooks={
  &forall,
  &token,
  &gram,
  &call,
  &xyz,
  &gcc,
  &header,
  &source,
  &mesh,
  &vars,
  &mains
};


// ****************************************************************************
// * kokkos
// ****************************************************************************
backendHooks* kokkos(nablaMain *nabla){
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    calls.parallel=&cilkCalls;
  
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    calls.parallel=&openMPCalls;
  
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    hooks.pragma=&icc;

  nabla->call=&calls;
  return &hooks;
  
}
