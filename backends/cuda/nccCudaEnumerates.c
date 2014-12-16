// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
#include "nabla.h"

void cudaDefineEnumerates(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
/*********************************************************\n\
 * Forward enumerates\n\
 *********************************************************/\n\
#define CUDA_INI_CELL_THREAD(tid) \\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if ( tid>=NABLA_NB_CELLS) return;\n\
\n\
#define CUDA_INI_CELL_THREAD_RETURN_REAL(tid) \\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if ( tid>=NABLA_NB_CELLS) return -1.0;\n\
\n\
#define CUDA_INI_NODE_THREAD(tid)\\\n\
  const register int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
  if ( tid>=NABLA_NB_NODES) return;\n\
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
