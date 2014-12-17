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

int host_cell_node[8*NABLA_NB_CELLS];
int host_node_cell[8*NABLA_NB_NODES];
int host_node_cell_corner[8*NABLA_NB_NODES];
int host_node_cell_and_corner[2*8*NABLA_NB_NODES];


__attribute__((unused)) static void verifConnectivity(void){
  printf("\nVérification des connectivité des noeuds");
  for(int n=0;n<NABLA_NB_NODES;n+=1){
    printf("\nFocusing on node %%d",n);
    for(int nc=0;nc<8;nc++){
      printf("\n\tnode_%%d knows cell %%d",n,host_node_cell[nc]);
      printf(", and node_%%d knows cell %%d",n,host_node_cell_and_corner[2*nc+0]);
    }
  }
}

__attribute__((unused)) static void verifCorners(void){
  printf("\nVérification des coins des noeuds");
  for(int n=0;n<NABLA_NB_NODES;n+=1){
    printf("\nFocusing on node %%d",n);
    for(int c=0,nc=8*n;c<8;c+=1,nc+=1){
      if (host_node_cell_corner[nc]==-1) continue;
      printf("\n\tnode_%%d is corner #%%d of cell %%d",n,host_node_cell_corner[nc],host_node_cell[nc]);
      //dbg(DBG_INI,", and node_%%d is corner #%%d of cell %%d",n,node_cell_and_corner[2*nc+1],node_cell_and_corner[2*nc+0]);
    }
  }
}

__attribute__((unused)) static void verifNextPrev(void){
 for (int i=0; i<NABLA_NB_CELLS; ++i) {
    printf("\nNext/Prev(X) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirX*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirX*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    printf("\nNext/Prev(Y) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirY*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirY*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    printf("\nNext/Prev(Z) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirZ*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirZ*NABLA_NB_CELLS+i]);
  }
}


static int comparNodeCell(const void *a, const void *b){
  return (*(int*)a)>(*(int*)b);
}

static int comparNodeCellAndCorner(const void *pa, const void *pb){
  int *a=(int*)pa;
  int *b=(int*)pb;
  return a[0]>b[0];
}

static void host_set_corners(void){
  int iCell=0;
  for(int iZ=0;iZ<NABLA_NB_CELLS_Z_AXIS;iZ++){
    for(int iY=0;iY<NABLA_NB_CELLS_Y_AXIS;iY++){
      for(int iX=0;iX<NABLA_NB_CELLS_X_AXIS;iX++,iCell+=1){
        const int node_bid=iX + iY*NABLA_NB_NODES_X_AXIS + iZ*NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;
        host_cell_node[0*NABLA_NB_CELLS+iCell] = node_bid;
        host_cell_node[1*NABLA_NB_CELLS+iCell] = node_bid + 1;
        host_cell_node[2*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS + 1;
        host_cell_node[3*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS + 0;
        host_cell_node[4*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;
        host_cell_node[5*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS +1 ;
        host_cell_node[6*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS + NABLA_NB_NODES_X_AXIS+1;
        host_cell_node[7*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS + NABLA_NB_NODES_X_AXIS;
      }
    }
  }

  for(int n=0;n<NABLA_NB_NODES;n+=1){
    for(int c=0;c<8;++c){
      host_node_cell[8*n+c]=-1;
      host_node_cell_corner[8*n+c]=-1;
      host_node_cell_and_corner[2*(8*n+c)+0]=-1;//cell
      host_node_cell_and_corner[2*(8*n+c)+1]=-1;//corner
    }
  }
  for(int c=0;c<NABLA_NB_CELLS;c+=1){
    //printf("\nFocusing on cells %%d",c);
    for(int n=0;n<8;n++){
      const int iNode = host_cell_node[n*NABLA_NB_CELLS+c];
      //printf("\n\tcell_%%d @%%d: pushs node %%d",c,n,iNode);
      // les 8 emplacements donnent l'offset jusqu'aux mailles
      // node_corner a une structure en 8*NABLA_NB_NODES
      host_node_cell[8*iNode+n]=c;
      host_node_cell_corner[8*iNode+n]=n;
      host_node_cell_and_corner[2*(8*iNode+n)+0]=c;//cell
      host_node_cell_and_corner[2*(8*iNode+n)+1]=n;//corner
    }
  }
  // On va maintenant trier les connectivités node->cell pour assurer l'associativité
  // void qsort(void *base, size_t nmemb, size_t size,
  //                          int (*compar)(const void *, const void *));
  for(int n=0;n<NABLA_NB_NODES;n+=1){
    qsort(&host_node_cell[8*n],8,sizeof(int),comparNodeCell);
    qsort(&host_node_cell_and_corner[2*8*n],8,2*sizeof(int),comparNodeCellAndCorner);
  }
  // And we come back to set our node_cell_corner
  for(int n=0;n<NABLA_NB_NODES;n+=1){
    for(int c=0;c<8;++c){
      host_node_cell[8*n+c]=host_node_cell_and_corner[2*(8*n+c)+0];
      host_node_cell_corner[8*n+c]=host_node_cell_and_corner[2*(8*n+c)+1];
    }
  }
  //verifCorners();
}


#define xOf7(n) (n%%NABLA_NB_CELLS_X_AXIS)
#define yOf7(n) ((n/NABLA_NB_CELLS_X_AXIS)%%NABLA_NB_CELLS_Y_AXIS)
#define zOf7(n) ((n/(NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS))%%NABLA_NB_CELLS_Z_AXIS)


/*__device__ void sort(int *crnr,int *cell, int n, int i){
  if (i==0) return;
  //if (n==31) printf("\ncorner for node %%d, i=%%d, cell=%%d, cell-1=%%d",n,i,cell[8*n+i],cell[8*n+i-1]);
  if (cell[8*n+i-1]>cell[8*n+i]) return;
  {
    const int tmpCell=cell[8*n+i-1];
    const int tmpCrnr=crnr[8*n+i-1];
    cell[8*n+i-1]=cell[8*n+i];
    cell[8*n+i]=tmpCell;
    crnr[8*n+i-1]=crnr[8*n+i];
    crnr[8*n+i]=tmpCrnr;
  }
  }*/
__global__ void nabla_set_next_prev(int *cell_node,
                                    int *cell_prev,
                                    int *cell_next,
                                    int *node_cell,
                                    int *node_cell_corner,
                                    int *node_cell_corner_idx){
  CUDA_INI_CELL_THREAD(tcid);
  const int c=tcid;
  /*const int iNode0 = cell_node[0*NABLA_NB_CELLS+c];
  const int iNode1 = cell_node[1*NABLA_NB_CELLS+c];
  const int iNode2 = cell_node[2*NABLA_NB_CELLS+c];
  const int iNode3 = cell_node[3*NABLA_NB_CELLS+c];
  const int iNode4 = cell_node[4*NABLA_NB_CELLS+c];
  const int iNode5 = cell_node[5*NABLA_NB_CELLS+c];
  const int iNode6 = cell_node[6*NABLA_NB_CELLS+c];
  const int iNode7 = cell_node[7*NABLA_NB_CELLS+c];
  //printf("\nCell #%%d has as nodes: %%d,%%d,%%d,%%d,%%d,%%d,%%d,%%d",c, iNode0,iNode1,iNode2,iNode3,iNode4,iNode5,iNode6,iNode7);
  node_cell[8*iNode0+0]=c;
  node_cell[8*iNode1+1]=c;
  node_cell[8*iNode2+2]=c;
  node_cell[8*iNode3+3]=c;
  node_cell[8*iNode4+4]=c;
  node_cell[8*iNode5+5]=c;
  node_cell[8*iNode6+6]=c;
  node_cell[8*iNode7+7]=c;*/
  /*
  node_cell_corner[8*iNode0+node_cell_corner_idx[iNode0]]=node_cell_corner_idx[iNode0];
  sort(node_cell_corner,node_cell,iNode0,node_cell_corner_idx[iNode0]);
  node_cell_corner_idx[iNode0]+=1;
  
  node_cell_corner[8*iNode1+node_cell_corner_idx[iNode1]]=node_cell_corner_idx[iNode1];
  sort(node_cell_corner,node_cell,iNode1,node_cell_corner_idx[iNode1]);
  node_cell_corner_idx[iNode1]+=1;
  
  node_cell_corner[8*iNode2+node_cell_corner_idx[iNode2]]=node_cell_corner_idx[iNode2];
  sort(node_cell_corner,node_cell,iNode2,node_cell_corner_idx[iNode2]);
  node_cell_corner_idx[iNode2]+=1;
  
  node_cell_corner[8*iNode3+node_cell_corner_idx[iNode3]]=node_cell_corner_idx[iNode3];
  sort(node_cell_corner,node_cell,iNode3,node_cell_corner_idx[iNode3]);
  node_cell_corner_idx[iNode3]+=1;
  
  node_cell_corner[8*iNode4+node_cell_corner_idx[iNode4]]=node_cell_corner_idx[iNode4];
  sort(node_cell_corner,node_cell,iNode4,node_cell_corner_idx[iNode4]);
  node_cell_corner_idx[iNode4]+=1;
  
  node_cell_corner[8*iNode5+node_cell_corner_idx[iNode5]]=node_cell_corner_idx[iNode5];
  sort(node_cell_corner,node_cell,iNode5,node_cell_corner_idx[iNode5]);
  node_cell_corner_idx[iNode5]+=1;
  
  node_cell_corner[8*iNode6+node_cell_corner_idx[iNode6]]=node_cell_corner_idx[iNode6];
  sort(node_cell_corner,node_cell,iNode6,node_cell_corner_idx[iNode6]);
  node_cell_corner_idx[iNode6]+=1;
  
  node_cell_corner[8*iNode7+node_cell_corner_idx[iNode7]]=node_cell_corner_idx[iNode7];
  sort(node_cell_corner,node_cell,iNode7,node_cell_corner_idx[iNode7]);
  node_cell_corner_idx[iNode7]+=1;
  */
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
