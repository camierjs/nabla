

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

__device__ void mpi_reduce_min_local(Real *global_min_array, Real what){
  local_min[threadIdx.x]=what;
  __syncthreads();
  reduceMin(local_min,threadIdx.x);
  if (threadIdx.x==0)
    global_min_array[blockIdx.x]=local_min[0];
}

__device__ Real mpi_reduce_min(/*int tid,*/ Real *global_min_array, Real what){
  double min=global_min_array[0];
  // First reduce at thread level  
  mpi_reduce_min_local(global_min_array,what);
  // Tous les threads font le boulot pour l'instant
  // car on ne retourne pas coté host pour le min des blocs
  for (int i=1; i<blockIdx.x; i++)
    MINEQ(min,global_min_array[i]);
  return min;
}


// Pour l'exit, on force le global_deltat à une valeure négative
__device__ void cuda_exit(real *global_deltat){
  *global_deltat=-1.0;
}
