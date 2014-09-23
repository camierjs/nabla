

// ****************************************************************************
// * ERROR HANDLING
// ****************************************************************************
static void HandleError( cudaError_t err,
                         const char *file,
                         int line,
                         int exit_status){
  if (err != cudaSuccess) {
    printf("\33[1;33m\t%%s in %%s at line %%d\33[m\n",
           cudaGetErrorString( err ), file, line);
    if (exit_status==EXIT_SUCCESS)
      printf("\33[1;33m\tThis error has been told not to be fatal here, exiting quietly!\33[m\n");
    exit( exit_status );
  }
}
#define CUDA_HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__, EXIT_FAILURE)) 
#define CUDA_HANDLE_ERROR_WITH_SUCCESS(err) (HandleError(err, __FILE__, __LINE__, EXIT_SUCCESS)) 
