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
                            NABLA_NB_NODES*sizeof(Real),                \
                            cudaMemcpyDeviceToHost));                   \
    for (int i=0; i<NABLA_NB_NODES; i+=1)                               \
      printf("\n\t#%%d %%.14f",i, host_node_##Variable[i]);             \
  }

#define dbgNodeVariableXYZDim0(Variable)                              \
  void dbgNodeVariableXYZDim0_##Variable(void){                       \
    if (!DBG_MODE) return;                                            \
    CUDA_HANDLE_ERROR(cudaMemcpy(&host_node_##Variable,                    \
                            node_##Variable,                          \
                            NABLA_NB_NODES*sizeof(Real3),             \
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
                          NABLA_NB_NODES*sizeof(Real3),
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
                            NABLA_NB_CELLS*sizeof(Real3),             \
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
                              8*sizeof(Real3),                        \
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
                            NABLA_NB_CELLS*sizeof(Real),                \
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
                              8*sizeof(Real),                         \
                              cudaMemcpyDeviceToHost));               \
    for(int i=0;i<8;i+=1)                                             \
      for (int c=0;c<NABLA_NB_CELLS;c+=1)                             \
        printf("\n\t%%s@%%d+#%%d=%%.14f]",                            \
               #Variable,                                             \
               i,c,                                                   \
               host_cell_##Variable[c][i]);                           \
  }
