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


// ****************************************************************************
// * CALLS
// ****************************************************************************
const nWhatWith cuHeaderTypedef[]={
  {"int","integer"},
  {"struct real3","Real3"},
  {"struct real3x3","Real3x3"},
  {NULL,NULL}
};

const nWhatWith cuHeaderDefines[]={
  {"Real3","real3"},
  {"Real","real"},
  {"ReduceMinToDouble(what)","reduce_min_kernel(global_device_shared_reduce_results,what)"},
  {"ReduceMaxToDouble(what)","reduce_max_kernel(global_device_shared_reduce_results,what)"},
  {"norm","fabs"},
  {"rabs","fabs"},
  {"square_root","sqrt"},
  {"cube_root","cbrt"},
  {"opAdd(u,v)", "(u+v)"},
  {"opSub(u,v)", "(u-v)"},
  {"opDiv(u,v)", "(u/v)"},
  {"opMul(u,v)", "(u*v)"},
  {"opScaMul(a,b)","dot3(a,b)"},
  {"opVecMul(a,b)","cross3(a,b)"},
  {"knAt(a)",""},
  {"fatal(a,b)","cudaThreadExit()"},
  {"synchronize(a)",""},
  {"reduce(how,what)","what"},
  {"xyz","int"},
  {"GlobalIteration", "*global_iteration"},
  {"PAD_DIV(nbytes, align)", "(((nbytes)+(align)-1)/(align))"},
  {"xs_cell_node(n)", "cell_node[tcid*NABLA_NODE_PER_CELL+n]"},
  {"xs_face_cell(c)", "face_cell[tfid+NABLA_NB_FACES*c]"},
  {NULL,NULL}
};

const char* cuHeaderForwards[]={
  "void gpuEnum(void);",
  NULL
};

const callHeader cudaHeader={
  cuHeaderForwards,
  cuHeaderDefines,
  cuHeaderTypedef
};

const static callSimd cudaSimdCalls={
  cuHookBits,
  cuHookGather,
  cuHookScatter,
  cuHookIncludes
};

calls cudaCalls={
  &cudaHeader,
  &cudaSimdCalls,
  NULL,
};

// ****************************************************************************
// * HOOKS
// ****************************************************************************
const hookXyz cuHookXyz={
  cuHookSysPrefix,
  cuHookPrevCell,
  cuHookNextCell,
  cuHookSysPostfix
};

const static hookPragma cuHookPragma={
  cuHookPragmaGccAlign
};

const static hookHeader cuHookHeader={
  cuHookHeaderDump,
  cuHookHeaderOpen,
  cuHookHeaderEnumerates,
  cuHookHeaderPrefix,
  cuHookHeaderIncludes,
  cuHookHeaderPostfix
};

// Hooks pour le source
const static hookSource cuHookSource={
  cuHookSourceOpen,
  cuHookSourceInclude
};
  
// Hooks pour le maillage
const static hookMesh cuHookMesh={
  cuHookMeshPrefix,
  cuHookMeshCore,
  cuHookMeshPostfix
};
  
// Hooks pour les variables
const static hookVars cuHookVars={
  cuHookVariablesInit,
  cuHookVariablesPrefix,
  cuHookVariablesMalloc,
  cuHookVariablesFree
};  

// Hooks pour le main
const static hookMain cuHookMain={
  cuHookMainPrefix,
  cuHookMainPreInit,
  cuHookMainVarInitKernel,
  cuHookMainVarInitCall,
  cuHookMainCore,
  cuHookMainPostInit,
  cuHookMainPostfix
};  

  
const static hookForAll cuHookForAll={
  cuHookForAllPrefix,
  cuHookForAllDump,
  cuHookForAllItem,
  cuHookForAllPostfix
};

const static hookToken cuHookToken={
  cuHookTokenPrefix, // prefix
  cuHookSwitchToken, // svvitch
  cuHookTurnTokenToVariable, // variable
  cuHookTurnTokenToOption, // option
  cuHookSystem, // system
  cuHookIteration,
  cuHookExit,
  cuHookTime,
  cuHookFatal,
  cuHookTurnBracketsToParentheses,
  cuHookIsTest,
  cuHookTokenPostfix
};

const static hookGrammar cuHookGrammar={
  cuHookFunction,
  cuHookJob,
  cuHookReduction,
  NULL, // primary_expression_to_return
  NULL, // returnFromArgument
  cudaHookDfsVariable
};
  
  
const static hookCall cuHookCall={
  cuHookAddCallNames,
  cuHookAddArguments,
  cuHookEntryPointPrefix,
  cuHookDfsForCalls,
  cuHookAddExtraParameters,
  cuHookDumpNablaParameterList
};


static hooks cuHooks={
  &cuHookForAll,
  &cuHookToken,
  &cuHookGrammar,
  &cuHookCall,
  &cuHookXyz,
  &cuHookPragma,
  &cuHookHeader,
  &cuHookSource,
  &cuHookMesh,
  &cuHookVars,
  &cuHookMain
};


// ****************************************************************************
// * nccCuda
// ****************************************************************************
hooks *cuda(nablaMain *nabla){
  nabla->call=&cudaCalls;
  return &cuHooks;
}
