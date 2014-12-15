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


// ****************************************************************************
// * ERROR HANDLING
// ****************************************************************************
static inline void HandleError( const cudaError_t err,
                                const char *file,
                                const int line,
                                const int exit_status){
  if (err != cudaSuccess) {
    printf("\33[1;33m\t%%s in file %%s at line %%d\33[m\n",
           cudaGetErrorString( err ), file, line);
    if (exit_status==EXIT_SUCCESS)
      printf("\33[1;33m\tThis error has been told not to be fatal here, exiting quietly!\33[m\n");
    exit( exit_status );
  }
}
#define CUDA_HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__, EXIT_FAILURE)) 
#define CUDA_HANDLE_ERROR_WITH_SUCCESS(err) (HandleError(err, __FILE__, __LINE__, EXIT_SUCCESS)) 


static inline void cudaCheckLastKernel(const char *errorMessage,
                                       const char *file,
                                       const int line){
  const cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err){
    printf("\33[1;33m\t%%s (error #%%i): cudaGetLastError() threw '%%s' in file %%d at line %%s.\n",
           errorMessage, (int)err, cudaGetErrorString(err),file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK_LAST_KERNEL(msg) (cudaCheckLastKernel(msg, __FILE__, __LINE__)) 
