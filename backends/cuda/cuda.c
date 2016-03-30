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
#include "nabla.tab.h"
#include "backends/cuda/cuda.h"

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
  {"BLOCKSIZE", "128"},
  {"CUDA_NB_THREADS_PER_BLOCK", "128"},
  {"WARP_BIT", "0"},
  {"WARP_SIZE", "1"},
  {"WARP_ALIGN", "8"}, 
  {"NABLA_NB_GLOBAL","1"},
  {"Real3","real3"},
  {"Real","real"},
  {"ReduceMinToDouble(what)","reduce_min_kernel(NABLA_NB_CELLS,global_device_shared_reduce_results,what)"},
  {"ReduceMaxToDouble(what)","reduce_max_kernel(NABLA_NB_CELLS,global_device_shared_reduce_results,what)"},
  {"norm","fabs"},
  {"rabs","fabs"},
  {"set(a)", "a"},
  {"square_root","sqrt"},
  {"cube_root","cbrt"},
  {"DBG_MODE", "(false)"},
  {"DBG_LVL", "(DBG_ALL)"},
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
  {"opScaMul(a,b)","dot3(a,b)"},
  {"opVecMul(a,b)","cross3(a,b)"},
  {"knAt(a)",""},
  {"fatal(a,b)","cudaThreadExit()"},
  {"synchronize(a)",""},
  {"reduce(how,what)","what"},
  {"xyz","int"},
  {"GlobalIteration", "*global_iteration"},
  {"MD_DirX","0"},
  {"MD_DirY","1"},
  {"MD_DirZ","2"},
  {"MD_Plus","0"},
  {"MD_Negt","4"},
  {"MD_Shift","3"},
  {"MD_Mask","7"}, // [sign,..]
  {"PAD_DIV(nbytes, align)", "(((nbytes)+(align)-1)/(align))"},
  {"xs_cell_node(n)", "xs_cell_node[c*NABLA_NODE_PER_CELL+n]"},
  {"xs_face_cell(c)", "xs_face_cell[f+NABLA_NB_FACES*c]"},
  {"xs_face_node(n)", "xs_face_node[f+NABLA_NB_FACES*n]"},
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
  NULL,
  xCallGather,
  xCallScatter,
  NULL
};

backendCalls cudaCalls={
  &cudaHeader,
  &cudaSimdCalls,
  NULL,
};

// ****************************************************************************
// * HOOKS
// ****************************************************************************
const hookXyz cuHookXyz={
  NULL,
  xHookPrevCell,
  xHookNextCell,
  xHookSysPostfix
};

const static hookPragma cuHookPragma={
  NULL 
};

const static hookHeader cuHookHeader={
  cuHookHeaderDump,
  xHookHeaderOpen,
  NULL,
  xHookHeaderPrefix,
  cuHookHeaderIncludes, // cuda_runtime
  cuHookHeaderPostfix
};

// Hooks pour le source
const static hookSource cuHookSource={
  cuHookSourceOpen, // .cu
  xHookSourceInclude,
  NULL
};
  
// Hooks pour le maillage
const static hookMesh cuHookMesh={
  xHookMeshPrefix, 
  xHookMeshCore,
  NULL
};
  
// Hooks pour les variables
const static hookVars cuHookVars={
  cuHookVariablesInit,
  cuHookVariablesPrefix,
  //xHookVariablesMalloc,
  cuHookVariablesMalloc,
  cuHookVariablesFree,
  NULL,
  NULL
};  

// Hooks pour le main
const static hookMain cuHookMain={
  cuHookMainPrefix,
  cuHookMainPreInit,
  NULL,
  xHookMainVarInitCall,
  cuHookMainCore,
  cuHookMainPostInit,
  cuHookMainPostfix
};  

  
const static hookForAll cuHookForAll={
  cuHookForAllPrefix, // tid continue
  NULL,
  NULL,
  NULL
};

const static hookToken cuHookToken={
  NULL,
  xHookSwitchToken,
  xHookTurnTokenToVariable,
  xHookTurnTokenToOption,
  xHookSystem,
  xHookIteration,
  cuHookExit, // cudaExit(global_deltat)
  xHookTime,
  xHookFatal,
  xHookTurnBracketsToParentheses,
  xHookIsTest,
  NULL
};

const static hookGrammar cuHookGrammar={
  NULL,
  NULL,
  cuHookReduction,
  NULL, // primary_expression_to_return
  NULL, // returnFromArgument
  xHookDfsVariable,
  NULL
};
  
  
const static hookCall cuHookCall={
  xHookAddCallNames,
  xHookAddArguments,
  cuHookEntryPointPrefix, // __device__ vs __global__
  xHookDfsForCalls,
  NULL,
  NULL
};


static hooks cuHooks={
  &cuHookForAll,
  &cuHookToken,
  &cuHookGrammar,
  &cuHookCall,
  &cuHookXyz,
  NULL,
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
