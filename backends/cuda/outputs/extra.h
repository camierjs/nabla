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
// * REDUCTIONS
// *****************************************************************************


// Tableau en shared memory pour contenir la reduction locale
__shared__ double shared_array[CUDA_NB_THREADS_PER_BLOCK];



// *****************************************************************************
// * Reduce MIN Kernel
// *****************************************************************************
__device__ double reduce_min_kernel(double *results, const Real what){
  const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  const unsigned int i = bid*blockDim.x+tid;
  int dualTid;
   
  // Le bloc dépose la valeure qu'il a
  //__syncthreads();
  shared_array[tid]=what;
  __syncthreads();

  for(int workers=blockDim.x>>1; workers>0; workers>>=1){
    // Seule la premiere moitié travaille
    if (tid >= workers) continue;
    dualTid = tid + workers;
    // On évite de piocher trop loin
    if (i >= NABLA_NB_CELLS) continue;
    if (dualTid >= NABLA_NB_CELLS) continue;
    if ((blockDim.x*bid + dualTid) >= NABLA_NB_CELLS) continue;
    // Voici ceux qui travaillent
    //printf("\n#%%03d/%%d of bloc #%%d <?= with #%%d", tid, workers, blockIdx.x, dualTid);
    // On évite de taper dans d'autres blocs
    //if (dualTid >= blockDim.x) continue;
    // ALORS on peut réduire:
    {
      const double tmp = shared_array[dualTid];
      //printf("\n#%%03d/%%d of bloc #%%d <?= with #%%d: %%.21e vs %%.21e",tid, workers, blockIdx.x, dualTid,shared_array[tid],shared_array[dualTid]);
      if (tmp < shared_array[tid])
        shared_array[tid] = tmp;
    }
    __syncthreads();
  }
  __syncthreads();
  if (tid==0){
    results[bid]=shared_array[0];
    //printf("\nBloc #%%d returned %%.21e", bid, results[bid]);
    __syncthreads();
  }
//#warning Fake return for now
  // There is still a reduction to do
  return results[bid];
}


__global__ void blockLevel(Real *results){
  //const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  //const unsigned int i = bid*blockDim.x + tid;
  if (tid==0){
    //printf("\nresults[0]=%%.21e, results[1]=%%.21e, results[2]=%%.21e", bid, results[0], results[1], results[2]);
  }
}

// *****************************************************************************
// * Pour l'exit, on force le global_deltat à une valeure négative
// *****************************************************************************
__device__ void cudaExit(Real *global_deltat){
  *global_deltat=-1.0;
}
