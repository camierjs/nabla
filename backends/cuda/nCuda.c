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
// * cudaInlines
// ****************************************************************************
void nCudaInlines(nablaMain *nabla){
  fprintf(nabla->entity->src,"#include \"%sEntity.h\"\n", nabla->entity->name);
}


// ****************************************************************************
// * cudaPragmas
// ****************************************************************************
char *nCudaPragmaGccIvdep(void){ return ""; }
char *nCudaPragmaGccAlign(void){ return "__align__(8)"; }



// ****************************************************************************
// *
//*****************************************************************************
static void cudaHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,
          "#ifndef __CUDA_%s_H__\n#define __CUDA_%s_H__",
          nabla->entity->name,nabla->entity->name);
}


// ****************************************************************************
// *
// ****************************************************************************
static void cudaHeaderIncludes(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,"\n\n\n\
// *****************************************************************************\n\
// * Includes\n\
// *****************************************************************************\n\
#include <iostream>\n\
#include <cstdio>\n\
#include <cstdlib>\n\
#include <sys/time.h>\n\
#include <stdlib.h>\n\
#include <stdio.h>\n\
#include <assert.h>\n\
#include <string.h>\n\
#include <vector>\n\
#include <math.h>\n\
#include <assert.h>\n\
#include <stdarg.h>\n\
#include <cuda_runtime.h>\n\
cudaError_t cudaCalloc(void **devPtr, size_t size){\n\
   if (cudaSuccess==cudaMalloc(devPtr,size))\n\
      return cudaMemset(*devPtr,0,size);\n\
   return cudaErrorMemoryAllocation;\n\
}\n");
//  #warning CUDA Cartesian here
  fprintf(nabla->entity->hdr,"\n\n\
// *****************************************************************************\n\
// * Cartesian stuffs\n\
// *****************************************************************************\n\
#define MD_DirX 0\n#define MD_DirY 1\n#define MD_DirZ 2\n\
//#warning empty libCartesianInitialize\n\
//__device__ void libCartesianInitialize(void){}\n");
}



static void cudaHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n#endif // __CUDA_%s_H__\n",nabla->entity->name);
}



/*****************************************************************************
 * nccCuda
 *****************************************************************************/
NABLA_STATUS nccCuda(nablaMain *nabla,
                   astNode *root,
                   const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];

  // Std Typedefs, Defines & Forwards
  const nHookHeader nCudaHeaderHooks={
    NULL, // dump
    NULL, // open
    NULL, // enums
    NULL, // prefix
    NULL, // include
    NULL  // postfix
  };
  
  const nCallSimd nCudaSimdCalls={
    nCudaHookBits,
    nCudaHookGather,
    nCudaHookScatter,
    nCudaHookIncludes
  };

  const nHookXyz nCudaXyzHooks={
    nCudaHookSysPrefix,
    nCudaHookPrevCell,
    nCudaHookNextCell,
    nCudaHookSysPostfix
  };
  
  const nHookForAll nCudaHookForAll={
    nCudaHookPrefixEnumerate,
    nCudaHookDumpEnumerate,
    nCudaHookItem,
    nCudaHookPostfixEnumerate
  };
  
  const nHookToken nCudaHookToken={
    nCudaHookTokenPrefix, // prefix
    nCudaHookSwitchToken, // svvitch
    nCudaHookTurnTokenToVariable, // variable
    nCudaHookTurnTokenToOption, // option
    nCudaHookSystem, // system
    nCudaHookIteration,
    nCudaHookExit,
    nCudaHookTime,
    nCudaHookFatal,
    nCudaHookTurnBracketsToParentheses,
    cudaHookIsTest,
    nCudaHookTokenPostfix
  };
  
  const nHookGrammar nCudaHookGrammar={
    nCudaHookFunction,
    nCudaHookJob,
    nCudaHookReduction,
    NULL, // primary_expression_to_return
    NULL // returnFromArgument
  };
  
  const nHookCall nCudaHookCall={
    nCudaHookAddCallNames,
    nCudaHookAddArguments,
    nCudaHookEntryPointPrefix,
    nCudaHookDfsForCalls,
    nCudaHookAddExtraParameters,
    nCudaHookDumpNablaParameterList
  };
  
  const nHookPragma cudaPragmaGCCHooks={
    //nCudaPragmaGccIvdep,
    nCudaPragmaGccAlign
  };

  nHooks nCudaBackendHooks={
    &nCudaHookForAll,
    &nCudaHookToken,
    &nCudaHookGrammar,
    &nCudaHookCall,
    &nCudaXyzHooks, // xyz
    &cudaPragmaGCCHooks, // pragma
    &nCudaHeaderHooks, // header
    NULL, // source
    NULL, // mesh
    NULL, // vars
    NULL // main
  };
  
  nCalls nCudaBackendCalls={
    NULL, // header
    &nCudaSimdCalls, // simd
    NULL, // parallel
  };
  
  nabla->call=&nCudaBackendCalls;
  nabla->hook=&nCudaBackendHooks;

  // Rajout de la variable globale 'iteration'
  nablaVariable *iteration = nMiddleVariableNew(nabla);
  nMiddleVariableAdd(nabla, iteration);
  iteration->axl_it=false;
  iteration->item=strdup("global");
  iteration->type=strdup("integer");
  iteration->name=strdup("iteration");
  
  // Rajout de la variable globale 'min_array'
  // Pour la réduction aux blocs
  nablaVariable *device_shared_reduce_results = nMiddleVariableNew(nabla);
  nMiddleVariableAdd(nabla, device_shared_reduce_results);
  device_shared_reduce_results->axl_it=false;
  device_shared_reduce_results->item=strdup("global");
  device_shared_reduce_results->type=strdup("real");
  device_shared_reduce_results->name=strdup("device_shared_reduce_results");

  // Ouverture du fichier source du entity
  sprintf(srcFileName, "%sEntity.cu", nabla->name);
  if ((nabla->entity->src=fopen(srcFileName, "w")) == NULL) exit(NABLA_ERROR);

  // Ouverture du fichier header du entity
  sprintf(hdrFileName, "%sEntity.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
  
  // Dump des includes dans le header file, puis des typedefs, defines, debug & errors stuff
  cudaHeaderPrefix(nabla);
  cudaHeaderIncludes(nabla);
  nMiddleTypedefs(nabla,nCudaHookTypedef);
  cudaHeaderHandleErrors(nabla);
  nMiddleDefines(nabla,nCudaHookDefines);
  nMiddleForwards(nabla,nCudaHookForwards);
  cudaDefineEnumerates(nabla);
   
  // Génération du maillage
  cudaMesh(nabla);
  cudaMeshConnectivity(nabla);

  // Rajout de la classe Real3 et des extras
  cudaHeaderDebug(nabla);
  cudaHeaderReal3(nabla);
  cudaHeaderExtra(nabla);
  cudaHeaderMesh(nabla);
  cudaHeaderItems(nabla);

  // Dump dans le fichier source
  nCudaInlines(nabla);
  nccCudaMainMeshConnectivity(nabla);
  
  // Parse du code préprocessé et lance les hooks associés
  nMiddleGrammar(root,nabla);
  nccCudaMainVarInitKernel(nabla);

  // Partie PREFIX
  nccCudaMainPrefix(nabla);
  cudaVariablesPrefix(nabla);
  nccCudaMainMeshPrefix(nabla);
  
  // Partie Pré Init
  nccCudaMainPreInit(nabla);
  nccCudaMainVarInitCall(nabla);
      
  // Dump des entry points dans le main
  nccCudaMain(nabla);

  // Partie Post Init
  nccCudaMainPostInit(nabla);
  
  // Partie POSTFIX
  cudaHeaderPostfix(nabla); 
  nccCudaMainMeshPostfix(nabla);
  cudaVariablesPostfix(nabla);
  nccCudaMainPostfix(nabla);
  return NABLA_OK;
}
