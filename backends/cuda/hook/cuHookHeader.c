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
#include "backends/cuda/cuda.h"

extern char* cuSparseHeader(nablaMain *);


// ****************************************************************************
// * cuHookHeaderOpen
// ****************************************************************************
void cuHookHeaderOpen(nablaMain *nabla){
  char hdrFileName[NABLA_MAX_FILE_NAME];
  sprintf(hdrFileName, "%sEntity.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
}



// ****************************************************************************
// *
// ****************************************************************************
void cuHookHeaderIncludes(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,"\n\n\n\
// *****************************************************************************\n\
// * Includes from nabla->simd->includes\n\
// *****************************************************************************\n\
%s\
// *****************************************************************************\n\
// * Standard CUDA Includes\n\
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
}\n\
// *****************************************************************************\n\
// * Includes from nabla->parallel->includes()\n\
// *****************************************************************************\n\
%s",
          nabla->call->simd->includes(),
          "");//nabla->call->parallel->includes());
  
  fprintf(nabla->entity->hdr,"\n\n\
// *****************************************************************************\n\
// * Cartesian stuffs\n\
// *****************************************************************************\n\
#define MD_DirX 0\n#define MD_DirY 1\n#define MD_DirZ 2\n\
//#warning empty libCartesianInitialize\n\
//__device__ void libCartesianInitialize(void){}\n");
  
//  #warning MiddlEnd call from here
  nMiddleDefines(nabla,nabla->call->header->defines);
  nMiddleTypedefs(nabla,nabla->call->header->typedefs);
  nMiddleForwards(nabla,nabla->call->header->forwards);
  // Si on a la librairie Aleph, on la dump
  if ((nabla->entity->libraries&(1<<with_aleph))!=0)
    fprintf(nabla->entity->hdr, "%s", cuSparseHeader(nabla));
}

// ****************************************************************************
// * cuHookHeaderDump
// ****************************************************************************
void cuHookHeaderDump(nablaMain *nabla){
  assert(nabla->entity->name);
  cuHeaderTypes(nabla);
  cuHeaderExtra(nabla);
  cuHeaderMeshs(nabla);
  cuHeaderError(nabla);
  cuHeaderItems(nabla);
}


// ****************************************************************************
// *
//*****************************************************************************
void cuHookHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,
          "#ifndef __CUDA_%s_H__\n#define __CUDA_%s_H__",
          nabla->entity->name,nabla->entity->name);
}


// ****************************************************************************
// * ENUMERATES Hooks
// ****************************************************************************
void cuHookHeaderEnumerates(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
/*********************************************************\n\
 * Forward enumerates\n\
 *********************************************************/\n\
#define CUDA_INI_CELL_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid>=NABLA_NB_CELLS) return;\n\
\n\
#define CUDA_INI_CELL_THREAD_RETURN_REAL(tid) \\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid>=NABLA_NB_CELLS) return -1.0;\n\
\n\
#define CUDA_INI_NODE_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid>=NABLA_NB_NODES) return;\n\
\n\
#define CUDA_INI_FACE_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid>=NABLA_NB_FACES) return;\n\
\n\
#define CUDA_INI_INNER_FACE_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid>=NABLA_NB_FACES_INNER) return;\n\
\n\
#define CUDA_INI_OUTER_FACE_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid<NABLA_NB_FACES_INNER) return;\\\n\
  if (tid>=(NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER)) return;\n\
\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<8;n+=1)\n\
\n\
#define FOR_EACH_CELL_WARP(c) \n\
\n\
#define FOR_EACH_NODE_WARP(n) \n\
\n\
#define CUDA_INI_FUNCTION_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid!=0) return;\n\
\n\
#define CUDA_LAUNCHER_FUNCTION_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid>=NABLA_NB_CELLS) return;\n");
}


// ****************************************************************************
// *
//*****************************************************************************
void cuHookHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n#endif // __CUDA_%s_H__\n",nabla->entity->name);
}

