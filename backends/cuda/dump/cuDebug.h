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
// * DEBUG MACRO and DEFINITIONS
// *****************************************************************************
#define	DBG_OFF          0x0000ul
#define	DBG_CELL_VOLUME  0x0001ul
#define	DBG_CELL_CQS     0x0002ul
#define	DBG_GTH          0x0004ul
#define	DBG_NODE_FORCE   0x0008ul
#define	DBG_INI_EOS      0x0010ul
#define	DBG_EOS          0x0020ul
#define  DBG_DENSITY      0x0040ul
#define  DBG_MOVE_NODE    0x0080ul
#define	DBG_INI		     0x0100ul
#define	DBG_INI_CELL     0x0200ul
#define	DBG_INI_NODE     0x0400ul
#define	DBG_LOOP 	     0x0800ul
#define	DBG_FUNC_IN      0x1000ul
#define	DBG_FUNC_OUT     0x2000ul
#define  DBG_VELOCITY     0x4000ul
#define  DBG_BOUNDARIES   0x8000ul
#define	DBG_ALL		     0xFFFFul

#define  DBG_MODE         (DBG_ALL)
#define  DBG_LVL          (DBG_CELL_VOLUME)


// *****************************************************************************
// * DEBUG functions
// *****************************************************************************
void dbg(const unsigned int flag, const char *format,...){
  if (!DBG_MODE) return;
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
    CUDA_HANDLE_ERROR(cudaMemcpy(&host_node_##Variable,                    \
                            node_##Variable,                          \
                            NABLA_NB_NODES*sizeof(real3),             \
                            cudaMemcpyDeviceToHost));                 \
    for (int i=0; i<NABLA_NB_NODES; i+=1)                             \
      printf("\n\t#%%d [%%.14f %%.14f %%.14f]", i,                    \
             host_node_##Variable[i].x,                               \
             host_node_##Variable[i].y,                               \
             host_node_##Variable[i].z);                              \
  }

/*void dbgCoords(void){
  if (!DBG_MODE) return; 
  printf("\ndbgCoords:");
  CUDA_HANDLE_ERROR(cudaMemcpy(&host_node_coord, node_coord,
                          NABLA_NB_NODES*sizeof(real3),
                          cudaMemcpyDeviceToHost));
  for (int i=0; i<NABLA_NB_NODES; i+=1){
    printf("\n\t#%%d [%%.14f,%%.14f,%%.14f]",i,
           host_node_coord[i].x,
           host_node_coord[i].y,
           host_node_coord[i].z);
  }
  }*/


// *****************************************************************************
// * MACRO CELL functions
// *****************************************************************************
/*void dbgCellNodes(void){
  if (!DBG_MODE) return; 
  printf("\ndbgCellNodes:");
  CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_node_0, cell_node_0, NABLA_NB_CELLS*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_node_1, cell_node_1, NABLA_NB_CELLS*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_node_2, cell_node_2, NABLA_NB_CELLS*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_node_3, cell_node_3, NABLA_NB_CELLS*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_node_4, cell_node_4, NABLA_NB_CELLS*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_node_5, cell_node_5, NABLA_NB_CELLS*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_node_6, cell_node_6, NABLA_NB_CELLS*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_HANDLE_ERROR(cudaMemcpy(&host_cell_node_7, cell_node_7, NABLA_NB_CELLS*sizeof(int), cudaMemcpyDeviceToHost));
  for (int i=0; i<NABLA_NB_CELLS; i+=1){
    printf("\n\t#%%d [%%d,%%d,%%d,%%d,%%d,%%d,%%d,%%d]",i,
           host_cell_node_0[i],
           host_cell_node_1[i],
           host_cell_node_2[i],
           host_cell_node_3[i],
           host_cell_node_4[i],
           host_cell_node_5[i],
           host_cell_node_6[i],
           host_cell_node_7[i]);
  }
  }*/

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
