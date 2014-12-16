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

// *****************************************************************************
// * EXTRA
// *****************************************************************************

__device__ inline int NearestPowerOf2(int n){
  int x = 1;
  if (!n) return n;
  while(x < n)
    x <<= 1;
  return x;
}

// Tableau en shared memory pour contenir la reduction locale
__shared__ double local_min[CUDA_NB_THREADS_PER_BLOCK];

__device__ Real cuda_reduce_min(const Real what){
  double temp;
  int  thread2;
  int nTotalThreads = min(NearestPowerOf2(blockDim.x),NABLA_NB_CELLS);
  local_min[threadIdx.x]=what;
  __syncthreads();
  while(nTotalThreads > 1){
    const int halfPoint = (nTotalThreads >> 1);
    if (threadIdx.x < halfPoint){
     thread2 = threadIdx.x + halfPoint;
     //printf("\ntid=%%d, nTotalThreads=%%d, halfPoint=%%d, thread2=%%d, blockDim.x=%%d", threadIdx.x, nTotalThreads, halfPoint, thread2, blockDim.x);
      // Skipping the fictious threads blockDim.x ... blockDim_2-1
      if (thread2 < blockDim.x){
        temp = local_min[thread2];
        //printf("\n\t%%.21e vs %%.21e",local_min[threadIdx.x],local_min[thread2]);
        if (temp < local_min[threadIdx.x]){
          local_min[threadIdx.x] = temp;
          //printf("\n\t new local_min=%%.21e",local_min[threadIdx.x]);
        }
      }
    }
    __syncthreads();
    nTotalThreads = halfPoint;
  }
  return local_min[0];
}


// *****************************************************************************
// * Pour l'exit, on force le global_deltat à une valeure négative
// *****************************************************************************
__device__ void cudaExit(Real *global_deltat){
  *global_deltat=-1.0;
}
