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
#include "backends/okina/call/call.h"
#include "backends/okina/hook/hook.h"


// ****************************************************************************
// * CALLS
// ****************************************************************************
const callHeader okinaHeaderStd={
  nOkinaStdForwards,
  nOkinaStdDefines,
  nOkinaStdTypedef
};
const callSimd okinaSimdStd={
  nOkinaStdBits,
  nOkinaStdGather,
  nOkinaStdScatter,
  nOkinaStdIncludes
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
  nOkinaSseIncludes
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
  nOkinaAvxIncludes
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
  nOkinaMicIncludes
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
  nOkinaHookEnumeratePrefix,
  nOkinaHookEnumerateDump,
  nOkinaHookItem,
  nOkinaHookEnumeratePostfix
};

const hookToken token={
  nOkinaHookTokenPrefix,
  nOkinaHookTokenSwitch,
  nOkinaHookVariablesTurnTokenToVariable,
  nOkinaHookTurnTokenToOption,
  nOkinaHookVariablesSystem,
  nOkinaHookIteration,
  nOkinaHookExit,
  nOkinaHookTime,
  nOkinaHookFatal,
  nOkinaHookVariablesTurnBracketsToParentheses,
  okinaHookIsTest,
  nOkinaHookTokenPostfix
};

const hookGrammar gram={
  nOkinaHookFunction,
  nOkinaHookJob,
  nOkinaHookReduction,
  nOkinaHookPrimaryExpressionToReturn,
  nOkinaHookReturnFromArgument,
  okinaHookDfsVariable
};

const hookCall call={
  nOkinaHookAddCallNames,
  nOkinaHookAddArguments,
  nOkinaHookEntryPointPrefix,
  nOkinaHookDfsForCalls,
  nOkinaHookParamsAddExtra,
  nOkinaHookParamsDumpList
};

const hookXyz xyzStd={
  nOkinaHookSysPrefix,
  nOkinaStdPrevCell,
  nOkinaStdNextCell,
  nOkinaHookSysPostfix
};
const hookXyz xyzSse={
  nOkinaHookSysPrefix,
  nOkinaSsePrevCell,
  nOkinaSseNextCell,
  nOkinaHookSysPostfix
};
const hookXyz xyzAvx={
  nOkinaHookSysPrefix,
  nOkinaAvxPrevCell,
  nOkinaAvxNextCell,
  nOkinaHookSysPostfix
};
const hookXyz xyzMic={ 
  nOkinaHookSysPrefix,
  nOkinaMicPrevCell,
  nOkinaMicNextCell,
  nOkinaHookSysPostfix
};

const hookPragma icc ={
  nOkinaPragmaIccAlign
};
const hookPragma gcc={
  nOkinaPragmaGccAlign
};

const hookHeader header={
  nOkinaHeaderDump,
  nOkinaHeaderOpen,
  nOkinaHeaderDefineEnumerates,
  nOkinaHeaderPrefix,
  nOkinaHeaderIncludes,
  nOkinaHeaderPostfix
  };

const static hookSource source={
  hookSourceOpen,
  hookSourceInclude
};

const static hookMesh mesh={
  nOkinaMeshPrefix,
  nOkinaMeshCore,
  nOkinaMeshPostfix
};

const static hookVars vars={
  nOkinaVariablesInit,
  nOkinaVariablesPrefix,
  nOkinaVariablesMalloc,
  nOkinaVariablesFree
};

const static hookMain mains={
  nOkinaMainPrefix,
  nOkinaMainPreInit,
  nOkinaMainVarInitKernel,
  nOkinaMainVarInitCall,
  nOkinaMainHLT,
  nOkinaMainPostInit,
  nOkinaMainPostfix
};

// Definition of Okina's Hooks
static hooks okinaHooks={
  &forall,
  &token,
  &gram,
  &call,
  &xyzStd,
  &gcc,
  &header,
  &source,
  &mesh,
  &vars,
  &mains
};


// ****************************************************************************
// * okina with animate
// ****************************************************************************
hooks* okina(nablaMain *nabla){
  nabla->call=&okinaCalls;

  // Call switch between STD, SSE, AVX, MIC
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE){
    nabla->call->simd=&okinaSimdSse;
    nabla->call->header=&okinaHeaderSse;
  }
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX){
    nabla->call->simd=&okinaSimdAvx;
    nabla->call->header=&okinaHeaderAvx;
  }
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2){
    nabla->call->simd=&okinaSimdAvx;
    nabla->call->header=&okinaHeaderAvx;
  }
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC){
    nabla->call->simd=&okinaSimdMic;
    nabla->call->header=&okinaHeaderMic;
  }
  
  // Call between parallel modes
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nabla->call->parallel=&okinaCilk;
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nabla->call->parallel=&okinaOpenMP;

   
  // Hook des directions
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
    nabla->hook->xyz=&xyzSse;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
    nabla->hook->xyz=&xyzAvx;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
    nabla->hook->xyz=&xyzAvx;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
    nabla->hook->xyz=&xyzMic;

  // Hook du header
  /*if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
    nabla->hook->header=&headerSse;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
    nabla->hook->header=&headerAvx;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
    nabla->hook->header=&headerAvx;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
  nabla->hook->header=&headerMic;*/

  // Hook between ICC or GCC pragmas
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nabla->hook->pragma=&icc;

  return &okinaHooks;
}


// ****************************************************************************
// * Old okina way
// ****************************************************************************
/*NABLA_STATUS oldOkina(nablaMain *nabla,
                   astNode *root,
                   const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];

  nabla->call=&okinaCalls;
  nabla->hook=&okinaHooks;

  // Switch between STD, SSE, AVX, MIC
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE){
    nabla->call->simd=&okinaSimdSse;
    nabla->call->header=&okinaHeaderSse;
  }
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX){
    nabla->call->simd=&okinaSimdAvx;
    nabla->call->header=&okinaHeaderAvx;
  }
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2){
    nabla->call->simd=&okinaSimdAvx;
    nabla->call->header=&okinaHeaderAvx;
  }
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC){
    nabla->call->simd=&okinaSimdMic;
    nabla->call->header=&okinaHeaderMic;
  }

  // Gestion des directions
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
    nabla->hook->xyz=&xyzSse;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
    nabla->hook->xyz=&xyzAvx;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
    nabla->hook->xyz=&xyzAvx;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
    nabla->hook->xyz=&xyzMic;

  // Gestion du header
  if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
    nabla->hook->header=&headerSse;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
    nabla->hook->header=&headerAvx;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
    nabla->hook->header=&headerAvx;  
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
    nabla->hook->header=&headerMic;

  // Switch between parallel modes
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nabla->call->parallel=&okinaCilk;
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nabla->call->parallel=&okinaOpenMP;

  // Switch between ICC or GCC pragmas
  // Par defaut, on met GCC
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nabla->hook->pragma=&icc;


  // Rajout de la variable globale 'iteration'
  
  // Dump dans le HEADER: includes, typedefs, defines, debug, maths & errors stuff
  nOkinaHeaderPrefix(nabla);
  nOkinaHeaderIncludes(nabla);
  nMiddleDefines(nabla,nabla->call->header->defines);
  nMiddleTypedefs(nabla,nabla->call->header->typedefs);
  nMiddleForwards(nabla,nabla->call->header->forwards);

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
  
  dbg("\n\t[nOkina]  Deleting kernel names");
  toolUnlinkKtemp(nabla->entity->jobs);
  return NABLA_OK;
}
*/
