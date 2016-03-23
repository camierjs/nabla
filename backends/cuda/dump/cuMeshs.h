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

// ****************************************************************************
// * Dumped from meshs.h
// ****************************************************************************
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
        xs_cell_prev[MD_DirX*NABLA_NB_CELLS+i],
        i,
        xs_cell_next[MD_DirX*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    printf("\nNext/Prev(Y) for cells %%d <- #%%d -> %%d: ",
        xs_cell_prev[MD_DirY*NABLA_NB_CELLS+i],
        i,
        xs_cell_next[MD_DirY*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    printf("\nNext/Prev(Z) for cells %%d <- #%%d -> %%d: ",
        xs_cell_prev[MD_DirZ*NABLA_NB_CELLS+i],
        i,
        xs_cell_next[MD_DirZ*NABLA_NB_CELLS+i]);
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

__global__ void nabla_set_next_prev(int *cell_node,
                                    int *cell_prev,
                                    int *cell_next,
                                    int *node_cell,
                                    int *node_cell_corner,
                                    int *node_cell_corner_idx){
  CUDA_INI_CELL_THREAD(c);
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
