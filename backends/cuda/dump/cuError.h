///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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
