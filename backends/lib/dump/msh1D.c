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
// * 1D
// ****************************************************************************

// ****************************************************************************
// * Connectivité cell->node
// ****************************************************************************
static void nabla_ini_cell_node(const nablaMesh msh,
                                int *cell_node){
  dbg(DBG_OFF,"\nOn associe à chaque maille ses noeuds");
  int iCell=0;
  for(int iX=0;iX<msh.NABLA_NB_CELLS_X_AXIS;iX++,iCell+=1){
    const int cell_uid=iX;
    const int node_bid=iX;
    dbg(DBG_OFF,"\n\tSetting cell #%%d, cell_uid=%%d, node_bid=%%d",
        iCell,cell_uid,node_bid);
    cell_node[0*msh.NABLA_NB_CELLS+iCell] = node_bid;
    cell_node[1*msh.NABLA_NB_CELLS+iCell] = node_bid + 1;
    dbg(DBG_OFF,"\n\tCell_%%d's nodes are %%d,%%d", iCell,
        cell_node[0*msh.NABLA_NB_CELLS+iCell],
        cell_node[1*msh.NABLA_NB_CELLS+iCell]);
  }
}

// ****************************************************************************
// * Vérification: Connectivité cell->next et cell->prev
// ****************************************************************************
__attribute__((unused)) static
void verifNextPrev(const nablaMesh msh,
                   int *cell_prev, int *cell_next){
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    dbg(DBG_OFF,"\nNext/Prev(X) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirX*msh.NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirX*msh.NABLA_NB_CELLS+i]);
  }
}

// ****************************************************************************
// * Connectivité cell->next et cell->prev
// ****************************************************************************
static void nabla_ini_cell_next_prev(const nablaMesh msh,
                                     int *cell_prev, int *cell_next){
  dbg(DBG_OFF,"\nOn associe à chaque maille ses next et prev");
  // On met des valeurs négatives pour rGatherAndZeroNegOnes
  // Dans la direction X
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirX*msh.NABLA_NB_CELLS+i] = i-1 ;
    cell_next[MD_DirX*msh.NABLA_NB_CELLS+i] = i+1 ;
  }
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    if ((i%%msh.NABLA_NB_CELLS_X_AXIS)==0){
      cell_prev[MD_DirX*msh.NABLA_NB_CELLS+i] = -33333333 ;
      cell_next[MD_DirX*msh.NABLA_NB_CELLS+i+msh.NABLA_NB_CELLS_X_AXIS-1] = -44444444 ;
    }
  }
  verifNextPrev(msh,cell_prev,cell_next);
}

// ****************************************************************************
// * qsort compare fonction for a Node and a Cell
// ****************************************************************************
static int comparNodeCell(const void *a, const void *b){
  return (*(int*)a)>(*(int*)b);
}

// ****************************************************************************
// * qsort compare fonction for a Node, Cell and Corner
// ****************************************************************************
static int comparNodeCellAndCorner(const void *pa, const void *pb){
  int *a=(int*)pa;
  int *b=(int*)pb;
  return a[0]>b[0];
}

// ****************************************************************************
// * Vérification: Connectivité node->cell et node->corner
// ****************************************************************************
__attribute__((unused)) static
void verifConnectivity(const nablaMesh msh,
                       int* node_cell, int* node_cell_and_corner){
  dbg(DBG_OFF,"\nVérification des connectivité des noeuds");
  FOR_EACH_NODE_MSH(n){
    dbg(DBG_OFF,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL_MSH(c){
      dbg(DBG_OFF,"\n\tnode_%%d knows cell %%d",n,node_cell[nc]);
      dbg(DBG_OFF,", and node_%%d knows cell %%d",n,node_cell_and_corner[2*nc+0]);
    }
  }
}

__attribute__((unused)) static
void verifCorners(const nablaMesh msh,
                  int* node_cell, int* node_cell_corner){
  dbg(DBG_OFF,"\nVérification des coins des noeuds");
  FOR_EACH_NODE_MSH(n){
    dbg(DBG_OFF,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL_MSH(c){
      if (node_cell_corner[nc]==-1) continue;
      dbg(DBG_OFF,"\n\tnode_%%d is corner #%%d of cell %%d",
          n,node_cell_corner[nc],node_cell[nc]);
      //dbg(DBG_OFF,", and node_%%d is corner #%%d of cell %%d",n,node_cell_and_corner[2*nc+1],node_cell_and_corner[2*nc+0]);
    }
  }
}

// ****************************************************************************
// * Connectivité node->cell et node->corner
// ****************************************************************************
static void nabla_ini_node_cell(const nablaMesh msh,
                                const int* cell_node,
                                int *node_cell,
                                int* node_cell_corner,
                                int* node_cell_and_corner){
  dbg(DBG_OFF,"\nMaintenant, on re-scan pour remplir la connectivité des noeuds et des coins");
  dbg(DBG_OFF,"\nOn flush le nombre de mailles attachées à ce noeud");
  for(int n=0;n<msh.NABLA_NB_NODES;n+=1){
    for(int c=0;c<2;++c){
      node_cell[2*n+c]=-1;
      node_cell_corner[2*n+c]=-1;
      node_cell_and_corner[2*(2*n+c)+0]=-1;//cell
      node_cell_and_corner[2*(2*n+c)+1]=-1;//corner
    }
  }  
  for(int c=0;c<msh.NABLA_NB_CELLS;c+=1){
    dbg(DBG_OFF,"\nFocusing on cells %%d",c);
    for(int n=0;n<2;n++){
      const int iNode = cell_node[n*msh.NABLA_NB_CELLS+c];
      dbg(DBG_OFF,"\n\tcell_%%d @%%d: pushs node %%d",c,n,iNode);
      // les 2 emplacements donnent l'offset jusqu'aux mailles
      // node_corner a une structure en 2*NABLA_NB_NODES
      node_cell[2*iNode+n]=c;
      node_cell_corner[2*iNode+n]=n;
      node_cell_and_corner[2*(2*iNode+n)+0]=c;//cell
      node_cell_and_corner[2*(2*iNode+n)+1]=n;//corner
    }
  }
  // On va maintenant trier les connectivités node->cell pour assurer l'associativité
  // void qsort(void *base, size_t nmemb, size_t size,
  //                          int (*compar)(const void *, const void *));
  for(int n=0;n<msh.NABLA_NB_NODES;n+=1){
    qsort(&node_cell[2*n],2,sizeof(int),comparNodeCell);
    qsort(&node_cell_and_corner[2*2*n],2,2*sizeof(int),comparNodeCellAndCorner);
  }
  // And we come back to set our node_cell_corner
  for(int n=0;n<msh.NABLA_NB_NODES;n+=1)
    for(int c=0;c<2;++c)
      node_cell_corner[2*n+c]=node_cell_and_corner[2*(2*n+c)+1];
  //verifConnectivity(node_cell,node_cell_and_corner);
  verifCorners(msh,node_cell,node_cell_corner);
}

// ****************************************************************************
// * xOf7 
// ****************************************************************************
static double xOf7(const nablaMesh msh,
                   const int n){
  return ((double)(n%%msh.NABLA_NB_NODES_X_AXIS))*msh.NABLA_NB_NODES_X_TICK;
}

// ****************************************************************************
// * Vérification des coordonnées
// ****************************************************************************
__attribute__((unused)) static
void verifCoords(const nablaMesh msh,
                 Real *node_coord){
  dbg(DBG_OFF,"\nVérification des coordonnés des noeuds");
  FOR_EACH_NODE_MSH(n){
    dbg(DBG_OFF,"\n%%d:",n);
    dbgReal(DBG_OFF,node_coord[n]);
  }
}

// ****************************************************************************
// * Initialisation des coordonnées
// ****************************************************************************
static void nabla_ini_node_coords(const nablaMesh msh,
                                  Real *node_coord){
  dbg(DBG_OFF,"\nasserting (NABLA_NB_CELLS %% 1)==0...");
  assert((msh.NABLA_NB_CELLS %% 1)==0);
    
  for(int iNode=0; iNode<msh.NABLA_NB_NODES; iNode+=1){
    const int n=iNode;
    Real x;
    x=set(xOf7(msh,n));
    // Là où l'on poke le retour de okinaSourceMeshAoS_vs_SoA
    node_coord[iNode]=Real(x);
    dbg(DBG_OFF,"\nSetting nodes-vector #%%d @", n);
    //dbgReal(DBG_OFF,node_coord[iNode]);
  }
  //verifCoords(msh,node_coord);

}


// ****************************************************************************
// * nabla_ini_connectivity
// ****************************************************************************
static void nabla_ini_connectivity(const nablaMesh msh,
                                   Real *node_coord,
                                   int *cell_node,
                                   int *cell_prev, int *cell_next,
                                   int* cell_face,
                                   int *node_cell,
                                   int* node_cell_corner,
                                   int* node_cell_and_corner,
                                   int* face_cell,
                                   int* face_node){
  nabla_ini_node_coords(msh,node_coord);
  nabla_ini_cell_node(msh,cell_node);
  nabla_ini_cell_next_prev(msh,cell_prev,cell_next);
  nabla_ini_node_cell(msh,
                      cell_node,
                      node_cell,
                      node_cell_corner,
                      node_cell_and_corner);
  // face_cell and face_node are malloc'ed but with size 0
  //assert(face_cell && face_node);
  dbg(DBG_OFF,"\nIni done");
}

