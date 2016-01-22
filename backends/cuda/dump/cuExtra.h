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


// *****************************************************************************
// * REDUCTIONS
// *****************************************************************************
// Tableau en shared memory pour contenir la reduction locale
__shared__ double shared_array[CUDA_NB_THREADS_PER_BLOCK];



// *****************************************************************************
// * Reduce MIN Kernel
// *****************************************************************************
__device__ void reduce_min_kernel(real  *results, const real what){
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
  //return results[bid];
}


// *****************************************************************************
// * Reduce MAX Kernel
// *****************************************************************************
__device__ void reduce_max_kernel(real  *results, const real what){
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
      if (tmp > shared_array[tid])
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
  //return results[bid];
}



__global__ void blockLevel(real *results){
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
__device__ void cudaExit(real *global_deltat){
  *global_deltat=-1.0;
}
