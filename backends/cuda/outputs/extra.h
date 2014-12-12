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
