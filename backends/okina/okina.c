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
////#include "backends/okina/okina.h"
#include "backends/okina/call/call.h"


// ****************************************************************************
// * ivdep/align ICC Pragmas
// ****************************************************************************
char *nOkinaPragmaIccAlign(void){return "__declspec(align(WARP_ALIGN)) ";}


// ****************************************************************************
// * ivdep/align  GCC Pragmas
// ****************************************************************************
char *nOkinaPragmaGccAlign(void){return "__attribute__ ((aligned(WARP_ALIGN))) ";}


// ****************************************************************************
// * CALLS
// ****************************************************************************
const callHeader okinaHeaderStd={
  nOkinaStdForwards, //xCallHeaderForwards,
  nOkinaStdDefines, //xCallHeaderDefines,
  nOkinaStdTypedef // xCallHeaderTypedef
};
const callSimd okinaSimdStd={
  nOkinaStdBits,
  xCallGather,
  xCallScatter,
  nOkinaStdIncludes,
  nOkinaStdUid
};

const callHeader okinaHeaderSse={
  nOkinaSseForwards,
  nOkinaSseDefines,
  nOkinaSseTypedef
};
const callSimd okinaSimdSse={
  nOkinaSseBits,
  nOkinaSseGather,
  nOkinaSseScatter,
  nOkinaSseIncludes,
  nOkinaSseUid
};

const callHeader okinaHeaderAvx={
  nOkinaAvxForwards,
  nOkinaAvxDefines,
  nOkinaAvxTypedef
};
const callSimd okinaSimdAvx={
  nOkinaAvxBits,
  nOkinaAvxGather,
  nOkinaAvxScatter,
  nOkinaAvxIncludes,
  nOkinaAvxUid
};

const callHeader okinaHeaderMic={
  nOkinaMicForwards,
  nOkinaMicDefines,
  nOkinaMicTypedef
};
const callSimd okinaSimdMic={ 
  nOkinaMicBits,
  nOkinaMicGather,
  nOkinaMicScatter,
  nOkinaMicIncludes,
  nOkinaMicUid
};

const callHeader okinaHeader512={
  nOkina512Forwards,
  nOkina512Defines,
  nOkina512Typedef
};
const callSimd okinaSimd512={ 
  nOkina512Bits,
  nOkina512Gather,
  nOkina512Scatter,
  nOkina512Includes,
  nOkina512Uid
};

const callParallel okinaCilk={
  nOkinaParallelCilkSync,
  nOkinaParallelCilkSpawn,
  nOkinaParallelCilkLoop,
  nOkinaParallelCilkIncludes
};

const callParallel okinaOpenMP={
  nOkinaParallelOpenMPSync,
  nOkinaParallelOpenMPSpawn,
  nOkinaParallelOpenMPLoop,
  nOkinaParallelOpenMPIncludes
};

const callParallel okinaVoid={
  nOkinaParallelVoidSync,
  nOkinaParallelVoidSpawn,
  nOkinaParallelVoidLoop,
  nOkinaParallelVoidIncludes
};

backendCalls okinaCalls={
  &okinaHeaderStd,
  &okinaSimdStd,
  &okinaVoid
};


// ****************************************************************************
// * HOOKS
// ****************************************************************************
const hookForAll forall={
  NULL,
  xHookForAllDump,
  xHookForAllItem,
  xHookForAllPostfix
};

const hookToken token={
  NULL,
  xHookSwitchToken,
  xHookTurnTokenToVariable,
  xHookTurnTokenToOption,
  xHookSystem,
  xHookIteration,
  xHookExit,
  xHookError,
  xHookTime,
  xHookFatal,
  xHookTurnBracketsToParentheses,
  xHookIsTest,
  NULL
};

const hookGrammar gram={
  NULL,
  NULL,
  xHookReduction,
  NULL,
  NULL,
  xHookDfsVariable,
  NULL,
  NULL,
  NULL,
  NULL
};

const hookCall call={
  xHookAddCallNames,
  xHookAddArguments,
  xHookEntryPointPrefix,
  xHookDfsForCalls,
  NULL,
  NULL
  };

const hookXyz xyzStd={
  NULL,
  xHookPrevCell,
  xHookNextCell,
  xHookSysPostfix
};
const hookXyz xyzSse={
  NULL,
  nOkinaSsePrevCell,
  nOkinaSseNextCell,
  xHookSysPostfix
};
const hookXyz xyzAvx={
  NULL,
  nOkinaAvxPrevCell,
  nOkinaAvxNextCell,
  xHookSysPostfix
};
const hookXyz xyzMic={ 
  NULL,
  nOkinaMicPrevCell,
  nOkinaMicNextCell,
  xHookSysPostfix
};
const hookXyz xyz512={ 
  NULL,
  nOkina512PrevCell,
  nOkina512NextCell,
  xHookSysPostfix
};

const hookPragma pragmaICC ={
  nOkinaPragmaIccAlign
};
const hookPragma pragmaGCC={
  nOkinaPragmaGccAlign
};

const hookHeader header={
  nOkinaHeaderDump,
  NULL,
  xHookHeaderOpen,
  xHookHeaderDefineEnumerates,
  xHookHeaderPrefix,
  xHookHeaderIncludes,
  xHookHeaderAlloc,
  xHookHeaderPostfix
  };

const static hookSource source={
  xHookSourceOpen,
  xHookSourceInclude,
  xHookSourceNamespace
};

const static hookMesh mesh={
  xHookMeshPrefix,
  xHookMeshCore,
  xHookMeshPostfix
};

const static hookVars vars={
  xHookVariablesInit,
  xHookVariablesPrefix,
  xHookVariablesMalloc,
  xHookVariablesFree,
  NULL,
  xHookVariablesODecl
};

const static hookMain mains={
  xHookMainPrefix,
  xHookMainPreInit,
  xHookMainVarInitKernel,
  xHookMainVarInitCall,
  xHookMainHLT,
  xHookMainPostInit,
  xHookMainPostfix
};

// Definition of Okina's Hooks
static hooks okinaHooks={
  &forall,
  &token,
  &gram,
  &call,
  &xyzStd,
  &pragmaGCC,
  &header,
  &source,
  &mesh,
  &vars,
  &mains
};

static hooks* mfem(nablaMain *nabla){
  nOkinaMfemDump(nabla);
  return NULL;
}

// ****************************************************************************
// * okina with animate
// ****************************************************************************
hooks* okina(nablaMain *nabla){
  nabla->call=&okinaCalls;
  
  if (nabla->option==BACKEND_OPTION_OKINA_MFEM)
    return mfem(nabla);
 
  // Call switch between STD, SSE, AVX, MIC
  if (nabla->option==BACKEND_OPTION_OKINA_SSE){
    nabla->call->simd=&okinaSimdSse;
    nabla->call->header=&okinaHeaderSse;
  }
  if (nabla->option==BACKEND_OPTION_OKINA_AVX){
    nabla->call->simd=&okinaSimdAvx;
    nabla->call->header=&okinaHeaderAvx;
  }
  if (nabla->option==BACKEND_OPTION_OKINA_AVX2){
    nabla->call->simd=&okinaSimdAvx;
    nabla->call->header=&okinaHeaderAvx;
  }
  if (nabla->option==BACKEND_OPTION_OKINA_MIC){
    nabla->call->simd=&okinaSimdMic;
    nabla->call->header=&okinaHeaderMic;
  }
  if (nabla->option==BACKEND_OPTION_OKINA_AVX512){
    nabla->call->simd=&okinaSimd512;
    nabla->call->header=&okinaHeader512;
  }
  
  // Call between parallel modes
  if (nabla->parallelism==BACKEND_PARALLELISM_CILK)
    nabla->call->parallel=&okinaCilk;
      
  if (nabla->parallelism==BACKEND_PARALLELISM_OMP)
    nabla->call->parallel=&okinaOpenMP;

  // Hook des directions
  if (nabla->option==BACKEND_OPTION_OKINA_SSE) okinaHooks.xyz=&xyzSse;  
  if (nabla->option==BACKEND_OPTION_OKINA_AVX) okinaHooks.xyz=&xyzAvx;  
  if (nabla->option==BACKEND_OPTION_OKINA_AVX2) okinaHooks.xyz=&xyzAvx;  
  if (nabla->option==BACKEND_OPTION_OKINA_MIC) okinaHooks.xyz=&xyzMic;
  if (nabla->option==BACKEND_OPTION_OKINA_AVX512) okinaHooks.xyz=&xyz512;

  // Hook between ICC or GCC pragmas
  if (nabla->compiler==BACKEND_COMPILER_ICC)
    okinaHooks.pragma=&pragmaICC;

  return &okinaHooks;
}
