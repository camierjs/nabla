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
#define MINEQ(a,b) (a)=(((a)<(b))?(a):(b))
 
__device__ void reduceMin(Real *sresult, const int threadID){
    /* If number of threads is not a power of two, first add the ones
       after the last power of two into the beginning. At most one of
       these conditionals will be true for a given NPOT block size. */
  if (CUDA_NB_THREADS_PER_BLOCK > 512 && CUDA_NB_THREADS_PER_BLOCK <= 1024){
    __syncthreads();
    if (threadID < CUDA_NB_THREADS_PER_BLOCK-512)
      MINEQ(sresult[threadID],sresult[threadID + 512]);
  }
  if (CUDA_NB_THREADS_PER_BLOCK > 256 && CUDA_NB_THREADS_PER_BLOCK < 512){
    __syncthreads();
    if (threadID < CUDA_NB_THREADS_PER_BLOCK-256)
      MINEQ(sresult[threadID],sresult[threadID + 256]);
  }
  if (CUDA_NB_THREADS_PER_BLOCK > 128 && CUDA_NB_THREADS_PER_BLOCK < 256){
    __syncthreads();
    if (threadID < CUDA_NB_THREADS_PER_BLOCK-128)
      MINEQ(sresult[threadID],sresult[threadID + 128]);
  }
  if (CUDA_NB_THREADS_PER_BLOCK > 64 && CUDA_NB_THREADS_PER_BLOCK < 128){
    __syncthreads();
    if (threadID < CUDA_NB_THREADS_PER_BLOCK-64)
      MINEQ(sresult[threadID],sresult[threadID + 64]);
  }
  if (CUDA_NB_THREADS_PER_BLOCK > 32 && CUDA_NB_THREADS_PER_BLOCK < 64){
    __syncthreads();
    if (threadID < CUDA_NB_THREADS_PER_BLOCK-32)
      MINEQ(sresult[threadID],sresult[threadID + 32]);
  }
  if (CUDA_NB_THREADS_PER_BLOCK > 16 && CUDA_NB_THREADS_PER_BLOCK < 32){
    __syncthreads();
    if (threadID < CUDA_NB_THREADS_PER_BLOCK-16)
      MINEQ(sresult[threadID],sresult[threadID + 16]);
  }
  if (CUDA_NB_THREADS_PER_BLOCK > 8 && CUDA_NB_THREADS_PER_BLOCK < 16){
    __syncthreads();
    if (threadID < CUDA_NB_THREADS_PER_BLOCK-8)
      MINEQ(sresult[threadID],sresult[threadID + 8]);
  }
  if (CUDA_NB_THREADS_PER_BLOCK > 4 && CUDA_NB_THREADS_PER_BLOCK < 8){
    __syncthreads();
    if (threadID < CUDA_NB_THREADS_PER_BLOCK-4)
      MINEQ(sresult[threadID],sresult[threadID + 4]);
  }
  if (CUDA_NB_THREADS_PER_BLOCK > 2 && CUDA_NB_THREADS_PER_BLOCK < 4){
    __syncthreads();
    if (threadID < CUDA_NB_THREADS_PER_BLOCK-2)
      MINEQ(sresult[threadID],sresult[threadID + 2]);
  }
  if (CUDA_NB_THREADS_PER_BLOCK >= 512){
    __syncthreads();
    if (threadID < 256)
      MINEQ(sresult[threadID],sresult[threadID + 256]);
  }
    if (CUDA_NB_THREADS_PER_BLOCK >= 256){
        __syncthreads();
        if (threadID < 128)
          MINEQ(sresult[threadID],sresult[threadID + 128]);
    }
    if (CUDA_NB_THREADS_PER_BLOCK >= 128){
      __syncthreads();
      if (threadID < 64)
        MINEQ(sresult[threadID],sresult[threadID + 64]);
    }
    __syncthreads();
    if (threadID < 32) {
      volatile Real *vol = sresult;
      if (CUDA_NB_THREADS_PER_BLOCK >= 64) MINEQ(vol[threadID],vol[threadID + 32]);
      if (CUDA_NB_THREADS_PER_BLOCK >= 32) MINEQ(vol[threadID],vol[threadID + 16]);
      if (CUDA_NB_THREADS_PER_BLOCK >= 16) MINEQ(vol[threadID],vol[threadID + 8]);
      if (CUDA_NB_THREADS_PER_BLOCK >= 8)  MINEQ(vol[threadID],vol[threadID + 4]);
      if (CUDA_NB_THREADS_PER_BLOCK >= 4)  MINEQ(vol[threadID],vol[threadID + 2]);
      if (CUDA_NB_THREADS_PER_BLOCK >= 2)  MINEQ(vol[threadID],vol[threadID + 1]);
    }
    __syncthreads();
}

// Tableau en shared memory pour contenir la reduction locale
__shared__ double local_min[CUDA_NB_THREADS_PER_BLOCK];

// *****************************************************************************
__device__ void cuda_reduce_min_local(Real *min_array, Real what){
  local_min[threadIdx.x]=what;
  __syncthreads();
  reduceMin(local_min,threadIdx.x);
  if (threadIdx.x==0)
    min_array[blockIdx.x]=local_min[0];
}

// *****************************************************************************
__device__ Real cuda_reduce_min(Real *min_array, Real what){
  double minimum=min_array[0];
  // First reduce at thread level  
  cuda_reduce_min_local(min_array,what);
  // Tous les threads font le boulot pour l'instant
  // car on ne retourne pas coté host pour le min des blocs
  for (int i=1; i<blockIdx.x; i++)
    MINEQ(minimum,min_array[i]);
  return minimum;
}


// Pour l'exit, on force le global_deltat à une valeure négative
__device__ void cudaExit(Real *global_deltat){
  *global_deltat=-1.0;
}
