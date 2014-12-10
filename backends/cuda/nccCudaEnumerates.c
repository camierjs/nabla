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

// int tid = blockIdx.x*CUDA_NB_CELLS_PER_BLOCK+threadIdx.x*CUDA_NB_CELLS_PER_THREAD;
//  int tid = blockIdx.x*CUDA_NB_NODES_PER_BLOCK+threadIdx.x*CUDA_NB_NODES_PER_THREAD;
//__attribute__((unused)) __shared__ double local_min[CUDA_NB_THREADS_PER_BLOCK];
//#define FOR_EACH_CELL(c) for(int c=0;c<NABLA_NB_CELLS;c+=1)
/*#define FOR_EACH_CELL_WARP_NODE(n)\\n\
  for(register int cn=WARP_SIZE*c+WARP_SIZE-1;cn>=WARP_SIZE*c;--cn)\\n\
    for(int n=8-1;n>=0;--n)\n\
*/
/*#define FOR_EACH_NODE_WARP_CELL(n,c)\\n\
  for(register int nc=WARP_SIZE*n+WARP_SIZE-1;nc>=WARP_SIZE*n;--nc)\\n\
    for(int c=node_cell[nc][0];c>0;--c)\n\
*/
//#define FOR_EACH_NODE(n) for(int n=NABLA_NB_NODES-1;n>=0;--n)
//#define FOR_EACH_CELL_WARP(c) for(int c=CUDA_NB_CELLS_PER_THREAD-1;c>=0;--c)
//#define FOR_EACH_NODE_WARP(n) for(int n=CUDA_NB_NODES_PER_THREAD-1;n>=0;--n)
//#define CUDA_INI_FUNCTION_THREAD(tid) const register __attribute__((unused)) int tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n");


void cudaDefineEnumerates(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\n\
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
#warning CUDA_INI_FUNCTION_THREAD is tied to CUDA_INI_CELL_THREAD\n\
#define CUDA_INI_FUNCTION_THREAD(tid) CUDA_INI_CELL_THREAD(tid)\\\n\
//  const register int # tid = blockDim.x*blockIdx.x + threadIdx.x;\\\n\
//  if (tid!=0) return;\n");
}
