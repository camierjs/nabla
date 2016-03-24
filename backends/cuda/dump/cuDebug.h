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
// * DEBUG functions
// *****************************************************************************
void dbg(const unsigned int flag, const char *format,...){
  if ((flag&DBG_LVL)==0) return;
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
  fflush(stdout);
}


// *****************************************************************************
// * MACRO NODE functions
// *****************************************************************************
#define dbgNodeVariableDim0(Variable)                                   \
  void dbgNodeVariableDim0_##Variable(void){                            \
    if (!DBG_MODE) return;                                              \
    CUDA_HANDLE_ERROR(cudaMemcpy(&host_node_##Variable,                      \
                            node_##Variable,                            \
                            NABLA_NB_NODES*sizeof(real),                \
                            cudaMemcpyDeviceToHost));                   \
    for (int i=0; i<NABLA_NB_NODES; i+=1)                               \
      printf("\n\t#%%d %%.14f",i, host_node_##Variable[i]);             \
  }

#define dbgNodeVariableXYZDim0(Variable)                              \
  void dbgNodeVariableXYZDim0_##Variable(void){                       \
    if (!DBG_MODE) return;                                            \
    CUDA_HANDLE_ERROR(cudaMemcpy(&host_node_##Variable,               \
                            node_##Variable,                          \
                            NABLA_NB_NODES*sizeof(real3),             \
                            cudaMemcpyDeviceToHost));                 \
    for (int i=0; i<NABLA_NB_NODES; i+=1)                             \
      printf("\n\t#%%d [%%.14f %%.14f %%.14f]", i,                    \
             host_node_##Variable[i].x,                               \
             host_node_##Variable[i].y,                               \
             host_node_##Variable[i].z);                              \
  }

void dbgReal3(const unsigned int flag,Real3 node_coord){}

#define dbgCellVariableXYZDim0(Variable)                              \
  void dbgCellVariableXYZDim0_##Variable(void){                       \
    if (!DBG_MODE) return;                                            \
    CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_##Variable,               \
                            cell_##Variable,                          \
                            NABLA_NB_CELLS*sizeof(real3),             \
                            cudaMemcpyDeviceToHost));                 \
    for (int i=0; i<NABLA_NB_CELLS; i+=1)                             \
      printf("\n\t#%%d [%%.14f %%.14f %%.14f]", i,                    \
             host_cell_##Variable[i].x,                               \
             host_cell_##Variable[i].y,                               \
             host_cell_##Variable[i].z);                              \
  }

 
#define dbgCellVariableXYZDim1(Variable)                              \
  void dbgCellVariableXYZDim1_##Variable(void){                       \
    if (!DBG_MODE) return;                                            \
    for(int i=0;i<NABLA_NB_CELLS;i+=1)                                \
      CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_##Variable[i],          \
                              &cell_##Variable[i*8],                  \
                              8*sizeof(real3),                        \
                              cudaMemcpyDeviceToHost));               \
    for(int i=0;i<8;i+=1)                                             \
      for (int c=0;c<NABLA_NB_CELLS;c+=1)                             \
        printf("\n\t%%s@%%d+#%%d=[%%.14f,%%.14f,%%.14f]",             \
               #Variable,                                             \
               i,c,                                                   \
               host_cell_##Variable[c][i].x,                          \
               host_cell_##Variable[c][i].y,                          \
               host_cell_##Variable[c][i].z);                         \
  }



#define dbgCellVariableDim0(Variable)                                   \
  void dbgCellVariableDim0_##Variable(void){                            \
    if (!DBG_MODE) return;                                              \
    CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_##Variable,                 \
                            cell_##Variable,                            \
                            NABLA_NB_CELLS*sizeof(real),                \
                            cudaMemcpyDeviceToHost));                   \
    for (int i=0; i<NABLA_NB_CELLS; i+=1)                               \
      printf("\n\t cell[%%d]=%%.14f",i, host_cell_##Variable[i]);       \
  }


#define dbgCellVariableDim1(Variable)                                 \
  void dbgCellVariableDim1_##Variable(void){                          \
    if (!DBG_MODE) return;                                            \
    for(int i=0;i<NABLA_NB_CELLS;i+=1)                                \
      CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_##Variable[i],          \
                              &cell_##Variable[i*8],                  \
                              8*sizeof(real),                         \
                              cudaMemcpyDeviceToHost));               \
    for(int i=0;i<8;i+=1)                                             \
      for (int c=0;c<NABLA_NB_CELLS;c+=1)                             \
        printf("\n\t%%s@%%d+#%%d=%%.14f]",                            \
               #Variable,                                             \
               i,c,                                                   \
               host_cell_##Variable[c][i]);                           \
  }
