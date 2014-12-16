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

#define xOf7(n) (n%%NABLA_NB_CELLS_X_AXIS)
#define yOf7(n) ((n/NABLA_NB_CELLS_X_AXIS)%%NABLA_NB_CELLS_Y_AXIS)
#define zOf7(n) ((n/(NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS))%%NABLA_NB_CELLS_Z_AXIS)

__global__ void nabla_rescan_connectivity(int *cell_node,
                                          int *cell_prev,
                                          int *cell_next,
                                          int *node_cell,
                                          int *node_cell_corner){
  CUDA_INI_CELL_THREAD(tcid);
  const int c=tcid;
  const int iNode0 = cell_node[0*NABLA_NB_CELLS+c];
  const int iNode1 = cell_node[1*NABLA_NB_CELLS+c];
  const int iNode2 = cell_node[2*NABLA_NB_CELLS+c];
  const int iNode3 = cell_node[3*NABLA_NB_CELLS+c];
  const int iNode4 = cell_node[4*NABLA_NB_CELLS+c];
  const int iNode5 = cell_node[5*NABLA_NB_CELLS+c];
  const int iNode6 = cell_node[6*NABLA_NB_CELLS+c];
  const int iNode7 = cell_node[7*NABLA_NB_CELLS+c];
  node_cell[8*iNode0+0]=c;
  node_cell[8*iNode1+1]=c;
  node_cell[8*iNode2+2]=c;
  node_cell[8*iNode3+3]=c;
  node_cell[8*iNode4+4]=c;
  node_cell[8*iNode5+5]=c;
  node_cell[8*iNode6+6]=c;
  node_cell[8*iNode7+7]=c;
  node_cell_corner[8*iNode0+0]=0;
  node_cell_corner[8*iNode1+1]=1;
  node_cell_corner[8*iNode2+2]=2;
  node_cell_corner[8*iNode3+3]=3;
  node_cell_corner[8*iNode4+4]=4;
  node_cell_corner[8*iNode5+5]=5;
  node_cell_corner[8*iNode6+6]=6;
  node_cell_corner[8*iNode7+7]=7;
  //printf("\nCell #%%d has as nodes: %%d,%%d,%%d,%%d,%%d,%%d,%%d,%%d",c, iNode0,iNode1,iNode2,iNode3, iNode4,iNode5,iNode6,iNode7);
  
  // On met des valeurs négatives afin que le gatherk_and_zero_neg_ones puisse les reconaitre
  {// Dans la direction X
    const int i=c;
    const int idx=MD_DirX*NABLA_NB_CELLS+i;
    //printf("\nNext/Prev(X) #%%d, xOf7=%%d",c,xOf7(i));
    cell_prev[idx] = i-1 ;
    cell_next[idx] = i+1 ;
    if ((i%%NABLA_NB_CELLS_X_AXIS)==0) cell_prev[idx] = -33333333;
    if (((i+1)%%NABLA_NB_CELLS_X_AXIS)==0) cell_next[idx] = -44444444;
    //printf("\nNext/Prev(X) for cells %%d <- #%%d -> %%d", cell_prev[idx],i,cell_next[idx]);
  }
  
  {// Dans la direction Y
    const int i=c;
    const int idy=MD_DirY*NABLA_NB_CELLS+i;
    //printf("\nNext/Prev(Y) #%%d, yOf7=%%d",c,yOf7(i));
    cell_prev[idy] = i-NABLA_NB_CELLS_X_AXIS ;
    cell_next[idy] = i+NABLA_NB_CELLS_X_AXIS ;
    if (yOf7(i)==0) cell_prev[idy] = -55555555;
    if (yOf7(i)==(NABLA_NB_CELLS_Y_AXIS-1)) cell_next[idy] = -66666666 ;
    //printf("\nNext/Prev(Y) for cells %%d <- #%%d -> %%d", cell_prev[idy], i, cell_next[idy]);
  }
  
  {// Dans la direction Z
    const int i=c;
    const int idz=MD_DirZ*NABLA_NB_CELLS+i;
    //printf("\nNext/Prev(Z) #%%d, zOf7=%%d",c,zOf7(i));
    cell_prev[idz] = i-NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS ;
    cell_next[idz] = i+NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS ;
    if (zOf7(i)==0) cell_prev[idz] = -77777777;
    if (zOf7(i)==(NABLA_NB_CELLS_Z_AXIS-1)) cell_next[idz] = -88888888 ;
    //printf("\nNext/Prev(Z) for cells %%d <- #%%d -> %%d", cell_prev[idz], i, cell_next[idz]);
  }
};
