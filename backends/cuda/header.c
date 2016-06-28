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
// * cuHookHeaderDump
// ****************************************************************************
void cuHookHeaderDump(nablaMain *nabla){
  assert(nabla->entity->name);
  cuHeaderTypes(nabla);
  cuHeaderExtra(nabla);
  cuHeaderError(nabla);
  cuHeaderDebug(nabla);
}


// ****************************************************************************
// * cuHookHeaderIncludes
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
#include <getopt.h>\n\
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
          nabla->call->simd->includes?nabla->call->simd->includes():"",
          "");//nabla->call->parallel->includes());
  
  nMiddleDefines(nabla,nabla->call->header->defines);
  nMiddleTypedefs(nabla,nabla->call->header->typedefs);
  nMiddleForwards(nabla,nabla->call->header->forwards);
  // Si on a la librairie Aleph, on la dump
  if ((nabla->entity->libraries&(1<<with_aleph))!=0)
    fprintf(nabla->entity->hdr, "%s", cuSparseHeader(nabla));
}



// ****************************************************************************
// * cuHeaderEnumerates
// ****************************************************************************
static void cuHeaderEnumerates(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// *****************************************************************************\n\
// * Forward enumerates\n\
// *****************************************************************************\n\
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
#define FOR_EACH_NODE(n) for(int n=0;n<NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<8;n+=1)\n\
#define FOR_EACH_NODE_CELL(c) for(int c=0,nc=NABLA_NODE_PER_CELL*n;c<NABLA_NODE_PER_CELL;c+=1,nc+=1)\n\
//#define FOR_EACH_NODE_CELL(c) for(int c=0;c<8;c+=1)\n\
\n\
//#define FOR_EACH_CELL_WARP(c) \n\
\n\
//#define FOR_EACH_NODE_WARP(n) \n\
\n\
#define CUDA_INI_FUNCTION_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid!=0) return;\n\
\n\
#define CUDA_LAUNCHER_FUNCTION_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if (tid>=NABLA_NB_CELLS) return;\n\
\n\
#define FOR_EACH_NODE_MSH(n) for(int n=0;n<msh.NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE_CELL_MSH(c)\
 for(int c=0,nc=msh.NABLA_NODE_PER_CELL*n;c<msh.NABLA_NODE_PER_CELL;c+=1,nc+=1)\n\n\
");
}


// ****************************************************************************
// * 
// ****************************************************************************
void cuHookHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ****************************************************************************\n\
// * nablaMshStruct\n\
// ****************************************************************************\n\
typedef struct nablaMshStruct{\n\
\tint NABLA_NODE_PER_CELL;\n\
\tint NABLA_CELL_PER_NODE;\n\
\tint NABLA_CELL_PER_FACE;\n\
\tint NABLA_NODE_PER_FACE;\n\
\tint NABLA_FACE_PER_CELL;\n\
\n\
\tint NABLA_NB_NODES_X_AXIS;\n\
\tint NABLA_NB_NODES_Y_AXIS;\n\
\tint NABLA_NB_NODES_Z_AXIS;\n\
\n\
\tint NABLA_NB_CELLS_X_AXIS;\n\
\tint NABLA_NB_CELLS_Y_AXIS;\n\
\tint NABLA_NB_CELLS_Z_AXIS;\n\
\n\
\tint NABLA_NB_FACES_X_INNER;\n\
\tint NABLA_NB_FACES_Y_INNER;\n\
\tint NABLA_NB_FACES_Z_INNER;\n\
\tint NABLA_NB_FACES_X_OUTER;\n\
\tint NABLA_NB_FACES_Y_OUTER;\n\
\tint NABLA_NB_FACES_Z_OUTER;\n\
\tint NABLA_NB_FACES_INNER;\n\
\tint NABLA_NB_FACES_OUTER;\n\
\tint NABLA_NB_FACES;\n\
\n\
\tdouble NABLA_NB_NODES_X_TICK;\n\
\tdouble NABLA_NB_NODES_Y_TICK;\n\
\tdouble NABLA_NB_NODES_Z_TICK;\n\
\n\
\tint NABLA_NB_NODES;\n\
\tint NABLA_NODES_PADDING;\n\
\tint NABLA_NB_CELLS;\n\
\tint NABLA_NB_NODES_WARP;\n\
\tint NABLA_NB_CELLS_WARP;\n\
}nablaMesh;\n");
  
  cuHeaderEnumerates(nabla);
  
  fprintf(nabla->entity->hdr,
          "\n\n#endif // __BACKEND_%s_H__\n",
          nabla->entity->name);
}

