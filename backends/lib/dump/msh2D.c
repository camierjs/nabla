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
// * 2D
// ****************************************************************************

// ****************************************************************************
// * Connectivité cell->node
// ****************************************************************************
static void nabla_ini_cell_node(const nablaMesh msh,
                                int *cell_node){
  if (DBG_DUMP) printf("\n[1;33mcell->node:[0m");
  dbg(DBG_INI_CELL,"\nOn associe a chaque maille ses noeuds");
  int iCell=0;
  for(int iY=0;iY<msh.NABLA_NB_CELLS_Y_AXIS;iY++){
    for(int iX=0;iX<msh.NABLA_NB_CELLS_X_AXIS;iX++,iCell+=1){
      const int cell_uid=iX + iY*msh.NABLA_NB_CELLS_X_AXIS;
      const int node_bid=iX + iY*msh.NABLA_NB_NODES_X_AXIS;
      dbg(DBG_INI_CELL,"\n\tSetting cell #%%d %%dx%%d, cell_uid=%%d, node_bid=%%d",
          iCell,iX,iY,cell_uid,node_bid);
      cell_node[0*msh.NABLA_NB_CELLS+iCell] = node_bid;
      cell_node[1*msh.NABLA_NB_CELLS+iCell] = node_bid + 1;
      cell_node[2*msh.NABLA_NB_CELLS+iCell] = node_bid + msh.NABLA_NB_NODES_X_AXIS + 1;
      cell_node[3*msh.NABLA_NB_CELLS+iCell] = node_bid + msh.NABLA_NB_NODES_X_AXIS + 0;
      dbg(DBG_INI_CELL,"\n\tCell_%%d's nodes are %%d,%%d,%%d,%%d", iCell,
          cell_node[0*msh.NABLA_NB_CELLS+iCell],
          cell_node[1*msh.NABLA_NB_CELLS+iCell],
          cell_node[2*msh.NABLA_NB_CELLS+iCell],
          cell_node[3*msh.NABLA_NB_CELLS+iCell]);
      if (DBG_DUMP) printf("\n\t%%d,%%d,%%d,%%d,",
            cell_node[0*msh.NABLA_NB_CELLS+iCell],
            cell_node[1*msh.NABLA_NB_CELLS+iCell],
            cell_node[2*msh.NABLA_NB_CELLS+iCell],
            cell_node[3*msh.NABLA_NB_CELLS+iCell]);
    }
  }
}
 
// ****************************************************************************
// * Vérification: Connectivité cell->next et cell->prev
// ****************************************************************************
__attribute__((unused)) static
void verifNextPrev(const nablaMesh msh,
                   int *cell_prev, int *cell_next){
 for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI_CELL,"\nNext/Prev(X) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirX*msh.NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirX*msh.NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI_CELL,"\nNext/Prev(Y) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirY*msh.NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirY*msh.NABLA_NB_CELLS+i]);
  }
}

// ****************************************************************************
// * Connectivité cell->next et cell->prev
// ****************************************************************************
static void nabla_ini_cell_next_prev(const nablaMesh msh,
                                     int *cell_prev, int *cell_next){
  dbg(DBG_INI_CELL,"\nOn associe a chaque maille ses next et prev");
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
  // Dans la direction Y
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirY*msh.NABLA_NB_CELLS+i] = i-msh.NABLA_NB_CELLS_X_AXIS ;
    cell_next[MD_DirY*msh.NABLA_NB_CELLS+i] = i+msh.NABLA_NB_CELLS_X_AXIS ;
  }
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    if ((i%%(msh.NABLA_NB_CELLS_X_AXIS*msh.NABLA_NB_CELLS_Y_AXIS))<msh.NABLA_NB_CELLS_Y_AXIS){
      cell_prev[MD_DirY*msh.NABLA_NB_CELLS+i] = -55555555 ;
      cell_next[MD_DirY*msh.NABLA_NB_CELLS+i+
                (msh.NABLA_NB_CELLS_X_AXIS-1)*msh.NABLA_NB_CELLS_Y_AXIS] = -66666666 ;
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
__attribute__((unused))
static void verifConnectivity(const nablaMesh msh,
                              int* node_cell,
                              int *node_cell_and_corner){
  if (DBG_DUMP) printf("\n[1;33mnode->cell:[0m");
  dbg(DBG_INI_NODE,"\nVérification des connectivité des noeuds");
  FOR_EACH_NODE_MSH(n){
    dbg(DBG_INI_NODE,"\nFocusing on node %%d",n);
    if (DBG_DUMP) printf("\n\t");
    FOR_EACH_NODE_CELL_MSH(c){
      dbg(DBG_INI_NODE,"\n\tnode_%%d knows cell %%d",n,node_cell[nc]);
      dbg(DBG_INI_NODE,", and node_%%d knows cell %%d",n,node_cell_and_corner[2*nc+0]);
      if (DBG_DUMP) printf(" %%d,",node_cell[nc]);
    }
  }
}

__attribute__((unused))
static void verifCorners(const nablaMesh msh,
                         int* node_cell,
                         int *node_cell_corner){
  if (DBG_DUMP) printf("\n[1;33mnode->corner:[0m");
  dbg(DBG_INI_NODE,"\nVérification des coins des noeuds");
  FOR_EACH_NODE_MSH(n){
    dbg(DBG_INI_NODE,"\nFocusing on node %%d",n);
    if (DBG_DUMP) printf("\n\t");
    FOR_EACH_NODE_CELL_MSH(c){
      //if (node_cell_corner[nc]==-1) continue;
      dbg(DBG_INI_NODE,"\n\tnode_%%d is corner #%%d of cell %%d",n,
          node_cell_corner[nc],node_cell[nc]);
      if (DBG_DUMP) printf(" %%d,",node_cell_corner[nc]);
      //dbg(DBG_INI,", and node_%%d is corner #%%d of cell %%d",n,node_cell_and_corner[2*nc+1],node_cell_and_corner[2*nc+0]);
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
  dbg(DBG_INI_NODE,"\nMaintenant, on re-scan pour remplir la connectivité des noeuds et des coins");
  dbg(DBG_INI_NODE,"\nOn flush le nombre de mailles attachées à ce noeud");
  for(int n=0;n<msh.NABLA_NB_NODES;n+=1){
    for(int c=0;c<msh.NABLA_CELL_PER_NODE;++c){
      node_cell[msh.NABLA_CELL_PER_NODE*n+c]=-1;
      node_cell_corner[msh.NABLA_CELL_PER_NODE*n+c]=-1;
      node_cell_and_corner[2*(msh.NABLA_CELL_PER_NODE*n+c)+0]=-1;//cell
      node_cell_and_corner[2*(msh.NABLA_CELL_PER_NODE*n+c)+1]=-1;//corner
    }
  }  
  for(int c=0;c<msh.NABLA_NB_CELLS;c+=1){
    dbg(DBG_INI_NODE,"\nFocusing on cell %%d",c);
    for(int n=0;n<msh.NABLA_CELL_PER_NODE;n++){
      const int iNode = cell_node[n*msh.NABLA_NB_CELLS+c];
      dbg(DBG_INI_NODE,"\n\tcell_%%d @%%d: pushs node %%d",c,n,iNode);
      // les NABLA_CELL_PER_NODE emplacements donnent l'offset jusqu'aux mailles
      // node_corner a une structure en NABLA_CELL_PER_NODE*NABLA_NB_NODES
      node_cell[msh.NABLA_CELL_PER_NODE*iNode+n]=c;
      node_cell_corner[msh.NABLA_CELL_PER_NODE*iNode+n]=n;
      node_cell_and_corner[2*(msh.NABLA_CELL_PER_NODE*iNode+n)+0]=c;//cell
      node_cell_and_corner[2*(msh.NABLA_CELL_PER_NODE*iNode+n)+1]=n;//corner
    }
  }
  // On va maintenant trier les connectivités node->cell pour assurer l'associativité
  // void qsort(void *base, size_t nmemb, size_t size,
  //            int (*compar)(const void *, const void *));
  for(int n=0;n<msh.NABLA_NB_NODES;n+=1){
    qsort(&node_cell[msh.NABLA_CELL_PER_NODE*n],
          msh.NABLA_CELL_PER_NODE,sizeof(int),comparNodeCell);
    qsort(&node_cell_and_corner[2*msh.NABLA_CELL_PER_NODE*n],
          msh.NABLA_CELL_PER_NODE,2*sizeof(int),comparNodeCellAndCorner);
  }
  // And we come back to set our node_cell_corner
  for(int n=0;n<msh.NABLA_NB_NODES;n+=1)
    for(int c=0;c<msh.NABLA_CELL_PER_NODE;++c)
      node_cell_corner[msh.NABLA_CELL_PER_NODE*n+c]=
        node_cell_and_corner[2*(msh.NABLA_CELL_PER_NODE*n+c)+1];
  verifConnectivity(msh,node_cell,node_cell_corner);
  verifCorners(msh,node_cell,node_cell_corner);
}

// ****************************************************************************
// * Connectivité face->cell
// * Un encodage des directions est utilisé en poids faibles:
// * bit:   2 1 0
// *     sign|.|.
// * MD_DirX => 1, MD_DirY => 2
// ****************************************************************************
__attribute__((unused)) static const int dir2bit(int d){
  assert((d==MD_DirX)||(d==MD_DirY));
  return d+1;
}
static const char cXY(int d){
  assert((d==MD_DirX)||(d==MD_DirY));
  return (d==MD_DirX)?'X':(d==MD_DirY)?'Y':'?';
}
static const char* sXY(int f){
  char str[32];
  const char sign=((f&4)==4)?'-':'+';
  f&=3;
  const char XorY=(f==1)?'X':(f==2)?'Y':'?';
  assert(XorY!='?');
  snprintf(str,32,"%%c%%c",sign,XorY);
  return strdup(str);
}
static char* f2d(int f,bool shift=true){
  char str[32];
  if (f>=0 && shift) snprintf(str,32,"%%d",f>>MD_Shift);
  if (f>=0 && !shift) snprintf(str,32,"%%d",f);
  if (f<0) snprintf(str,32,"[1;31m%%s[m",sXY(-f));
  return strdup(str);
}
static int nabla_ini_face_cell_outer_minus(const nablaMesh msh,
                                           int* face_cell,
                                           const int *iof,
                                           const int c,
                                           const int i,
                                           const int MD_Dir){
  const int f=iof[1];
  face_cell[0*msh.NABLA_NB_FACES+f] = (c<<MD_Shift)|MD_Negt|(MD_Dir+1);
  face_cell[1*msh.NABLA_NB_FACES+f] = -(MD_Negt|(MD_Dir+1));
  dbg(DBG_INI_FACE," %%s-%%c->%%s",
      f2d(face_cell[0*msh.NABLA_NB_FACES+f]),
      cXY(MD_Dir),
      f2d(face_cell[1*msh.NABLA_NB_FACES+f]));
  return 1;
}
static int nabla_ini_face_cell_inner(const nablaMesh msh,
                                     int* face_cell,
                                     const int *iof, const int c,
                                     const int i, const int MD_Dir){
  const int f=iof[0];
  face_cell[0*msh.NABLA_NB_FACES+f] = (c<<MD_Shift)|MD_Plus|(MD_Dir+1);
  if (MD_Dir==MD_DirX) face_cell[1*msh.NABLA_NB_FACES+f] = c+1;
  if (MD_Dir==MD_DirY) face_cell[1*msh.NABLA_NB_FACES+f] = c+msh.NABLA_NB_CELLS_X_AXIS;
  face_cell[1*msh.NABLA_NB_FACES+f] <<= MD_Shift;
  face_cell[1*msh.NABLA_NB_FACES+f] |= (MD_Plus|(MD_Dir+1));
  dbg(DBG_INI_FACE," %%s-%%c->%%s",
      f2d(face_cell[0*msh.NABLA_NB_FACES+f]),
      cXY(MD_Dir),
      f2d(face_cell[1*msh.NABLA_NB_FACES+f]));
  return 1;
}
static int nabla_ini_face_cell_outer_plus(const nablaMesh msh,
                                          int* face_cell,
                                          const int *iof, const int c,
                                          const int i, const int MD_Dir){
  const int f=iof[1];
  face_cell[0*msh.NABLA_NB_FACES+f] = (c<<MD_Shift)|MD_Plus|(MD_Dir+1);
  face_cell[1*msh.NABLA_NB_FACES+f] = -(MD_Plus|(MD_Dir+1));
  dbg(DBG_INI_FACE," %%s-%%c->%%s",
      f2d(face_cell[0*msh.NABLA_NB_FACES+f]),
      cXY(MD_Dir),
      f2d(face_cell[1*msh.NABLA_NB_FACES+f]));
  return 1;
}
static void nabla_ini_face_cell_XY(const nablaMesh msh,
                                   int* face_cell,
                                   int *f, const int c,
                                   const int i, const int MD_Dir){
  const int n =
    (MD_Dir==MD_DirX)?msh.NABLA_NB_CELLS_X_AXIS:
    (MD_Dir==MD_DirY)?msh.NABLA_NB_CELLS_Y_AXIS:-0xDEADBEEF;  
  if (i<n-1)  f[0]+=nabla_ini_face_cell_inner(msh,face_cell,f,c,i,MD_Dir);
  if (i==0)   f[1]+=nabla_ini_face_cell_outer_minus(msh,face_cell,f,c,i,MD_Dir);
  if (i==n-1) f[1]+=nabla_ini_face_cell_outer_plus(msh,face_cell,f,c,i,MD_Dir);
}
static void nabla_ini_face_cell(const nablaMesh msh,
                                int* face_cell){
  dbg(DBG_INI_FACE,"\n[1;33m[nabla_ini_face_cell] On associe a chaque maille ses faces:[m");
  int f[2]={0,msh.NABLA_NB_FACES_INNER}; // inner and outer faces
  for(int iY=0;iY<msh.NABLA_NB_CELLS_Y_AXIS;iY++){
    for(int iX=0;iX<msh.NABLA_NB_CELLS_X_AXIS;iX++){
      const int c=iX + iY*msh.NABLA_NB_CELLS_X_AXIS;
      dbg(DBG_INI_FACE,"\n\tCell #[1;36m%%d[m @ %%dx%%d:",c,iX,iY);
      nabla_ini_face_cell_XY(msh,face_cell,f,c,iX,MD_DirX);
      nabla_ini_face_cell_XY(msh,face_cell,f,c,iY,MD_DirY);
    }
  }
  dbg(DBG_INI_FACE,"\n\tNumber of faces = %%d",f[0]+f[1]-msh.NABLA_NB_FACES_INNER);
  assert(f[0]==msh.NABLA_NB_FACES_INNER);
  assert(f[1]==msh.NABLA_NB_FACES_INNER+msh.NABLA_NB_FACES_OUTER);
  assert((f[0]+f[1])==msh.NABLA_NB_FACES+msh.NABLA_NB_FACES_INNER);
  // On laisse les faces shiftées/encodées avec les directions pour les face_node
}

// ****************************************************************************
// * On les a shifté pour connaitre les directions, on flush les positifs
// ****************************************************************************
void nabla_ini_shift_back_face_cell(const nablaMesh msh,
                                    int* face_cell){
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1){
    if (face_cell[0*msh.NABLA_NB_FACES+f]>0) face_cell[0*msh.NABLA_NB_FACES+f]>>=MD_Shift;
    if (face_cell[1*msh.NABLA_NB_FACES+f]>0) face_cell[1*msh.NABLA_NB_FACES+f]>>=MD_Shift;
  }
  dbg(DBG_INI_FACE,"\n[nabla_ini_shift_back_face_cell] Inner faces:\n");
  for(int f=0;f<msh.NABLA_NB_FACES_INNER;f+=1)
    dbg(DBG_INI_FACE," %%s->%%s",
        f2d(face_cell[0*msh.NABLA_NB_FACES+f],false),
        f2d(face_cell[1*msh.NABLA_NB_FACES+f],false));
  dbg(DBG_INI_FACE,"\n[nabla_ini_shift_back_face_cell] Outer faces:\n");
  for(int f=msh.NABLA_NB_FACES_INNER;f<msh.NABLA_NB_FACES_INNER+msh.NABLA_NB_FACES_OUTER;f+=1)
    dbg(DBG_INI_FACE," %%s->%%s",
        f2d(face_cell[0*msh.NABLA_NB_FACES+f],false),
        f2d(face_cell[1*msh.NABLA_NB_FACES+f],false));
  dbg(DBG_INI_FACE,"\n[nabla_ini_shift_back_face_cell] All faces:\n");
  if (DBG_DUMP) printf("\n[1;33mface->cell:[0m");
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1){
    dbg(DBG_INI_FACE," %%d->%%d",
        face_cell[0*msh.NABLA_NB_FACES+f],
        face_cell[1*msh.NABLA_NB_FACES+f]);
    if (DBG_DUMP) printf("\n\t%%d,%%d,",
        face_cell[0*msh.NABLA_NB_FACES+f],
        face_cell[1*msh.NABLA_NB_FACES+f]);
  }
}

// ****************************************************************************
// * Connectivité cell->face
// ****************************************************************************
/*static void addThisfaceToCellConnectivity(const nablaMesh msh,
                                          int* cell_face,
                                          const int f, const int c){
  dbg(DBG_INI_FACE,"\n\t\t[addThisfaceToCellConnectivity] Adding face #%%d to cell %%d ",f,c);
  for(int i=0;i<msh.NABLA_FACE_PER_CELL;i+=1){
    // On scrute le premier emplacement 
    if (cell_face[i*msh.NABLA_NB_CELLS+c]>=0) continue;
    dbg(DBG_INI_FACE,"[%%d] ",i);
    cell_face[i*msh.NABLA_NB_CELLS+c]=f;
    break; // We're finished here
  }
  }*/
static void addThisfaceToCellConnectivity(const nablaMesh msh,
                                          const int* cell_node,const int c,
                                          const int* face_node,const int f,
                                          int* cell_face){
  const int n0 = face_node[0*msh.NABLA_NB_FACES+f];
  const int n1 = face_node[1*msh.NABLA_NB_FACES+f];
  dbg(DBG_INI_FACE,"\n\t\t[addThisfaceToCellConnectivity] Adding face #%%d (%%d->%%d) to cell %%d ",f,n0,n1,c);
  for(int i=0;i<msh.NABLA_FACE_PER_CELL;i+=1){
    const int cn0=cell_node[i*msh.NABLA_NB_CELLS+c];
    const int cn1=cell_node[((i+1)%%NABLA_NODE_PER_CELL)*msh.NABLA_NB_CELLS+c];
    const int ocn0=min(cn0,cn1);
    const int ocn1=max(cn0,cn1);
    dbg(DBG_INI_FACE,", cell_nodes %%d->%%d",ocn0,ocn1); 
    if (ocn0!=n0) continue;
    if (ocn1!=n1) continue;
    dbg(DBG_INI_FACE," at %%d",i);
    cell_face[i*msh.NABLA_NB_CELLS+c]=f;
    return; // We're finished here
  }
}


static void nabla_ini_cell_face(const nablaMesh msh,
                                const int* face_cell,
                                const int* cell_node,
                                const int* face_node,
                                int* cell_face){
  dbg(DBG_INI_FACE,"\n[1;33mOn revient pour remplir cell->face:[m (flushing)");
  for(int c=0;c<msh.NABLA_NB_CELLS;c+=1){
    for(int f=0;f<msh.NABLA_FACE_PER_CELL;f+=1){
      cell_face[f*msh.NABLA_NB_CELLS+c]=-1;
    }
  }
 
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1){
    const int cell0 = face_cell[0*msh.NABLA_NB_FACES+f];
    const int cell1 = face_cell[1*msh.NABLA_NB_FACES+f];
    dbg(DBG_INI_FACE,"\n\t[nabla_ini_cell_face] Pushing face #%%d: %%d->%%d",f,cell0,cell1);
    if (cell0>=0) addThisfaceToCellConnectivity(msh,cell_node,cell0,face_node,f,cell_face);
    if (cell1>=0) addThisfaceToCellConnectivity(msh,cell_node,cell1,face_node,f,cell_face);
  }
  
  dbg(DBG_INI_FACE,"\n[1;33mOn revient pour dumper cell->face:[m");
  if (DBG_DUMP) printf("\n[1;33mcell->face:[0m");
  for(int c=0;c<msh.NABLA_NB_CELLS;c+=1){
    if (DBG_DUMP) printf("\n");
    for(int f=0;f<msh.NABLA_FACE_PER_CELL;f+=1){
      if (DBG_DUMP) printf(" %%d,",cell_face[f*msh.NABLA_NB_CELLS+c]);
      if (cell_face[f*msh.NABLA_NB_CELLS+c]<0) continue;
      dbg(DBG_INI_FACE,"\n\t[nabla_ini_cell_face] cell[%%d]_face[%%d] %%d",
          c,f,cell_face[f*msh.NABLA_NB_CELLS+c]);
    }
  }
}


// ****************************************************************************
// * Connectivité face->node
// ****************************************************************************
static const char* i2XY(const int i){
  const int c=i&MD_Mask;
  //dbg(DBG_INI_FACE,"\n\t\t[33m[i2XY] i=%%d, Shift=%%d, c=%%d[m",i,MD_Shift,c);
  if (c==(MD_Plus|(MD_DirX+1))) return strdup("x+");
  if (c==(MD_Negt|(MD_DirX+1))) return strdup("x-");
  if (c==(MD_Plus|(MD_DirY+1))) return strdup("y+");
  if (c==(MD_Negt|(MD_DirY+1))) return strdup("y-");
  fprintf(stderr,"[i2XY] Error, could not distinguish XY!");
  exit(-1);
  return NULL;
}
static const char* c2XY(const int c){
  char str[16];
  if (c==-(MD_Negt|(MD_DirX+1))) return strdup("[1;31mX-[m");
  if (c==-(MD_Plus|(MD_DirX+1))) return strdup("[1;31mX+[m");
  if (c==-(MD_Negt|(MD_DirY+1))) return strdup("[1;31mY-[m");
  if (c==-(MD_Plus|(MD_DirY+1))) return strdup("[1;31mY+[m");
  if (snprintf(str,16,"%%d%%s",c>>MD_Shift,i2XY(c))<0) fprintf(stderr,"c2XY!");
  return strdup(str);
}
static void setFWithTheseNodes(const nablaMesh msh,
                               int* face_node,
                               int* cell_node,
                               const int f, const int c,
                               const int n0, const int n1){
  //dbg(DBG_INI_FACE,"\n[1;33m[setFWithTheseNodes] %%d & %%d[m",n0,n1);
  const int nid0=cell_node[n0*msh.NABLA_NB_CELLS+c];
  const int nid1=cell_node[n1*msh.NABLA_NB_CELLS+c];
  face_node[0*msh.NABLA_NB_FACES+f]=min(nid0,nid1);
  face_node[1*msh.NABLA_NB_FACES+f]=max(nid0,nid1);
}
static void nabla_ini_face_node(const nablaMesh msh,
                                const int* face_cell,
                                int* face_node,
                                int* cell_node){
  dbg(DBG_INI_FACE,"\n[1;33m[nabla_ini_face_node] On associe a chaque faces ses noeuds:[m");
  // On flush toutes les connectivités face_noeuds
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1)
    for(int n=0;n<msh.NABLA_NODE_PER_FACE;n+=1)
      face_node[n*msh.NABLA_NB_FACES+f]=-1;
  
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1){
    const int backCell=face_cell[0*msh.NABLA_NB_FACES+f];
    const int frontCell=face_cell[1*msh.NABLA_NB_FACES+f];
    dbg(DBG_INI_FACE,"\n\tFace #[1;36m%%d[m: %%d => %%d, ",f, backCell>>MD_Shift, frontCell>>MD_Shift);
    dbg(DBG_INI_FACE,"\t%%s => %%s: ", c2XY(backCell), c2XY(frontCell));
    // On va travailler avec sa backCell
    const int c=backCell>>MD_Shift;
    const int d=backCell &MD_Mask;
    dbg(DBG_INI_FACE,"\t%%d ", c);
    dbg(DBG_INI_FACE,"\t%%d ", d);
    assert(c>=0);
    if (d==(MD_Plus|(MD_DirX+1)))
      { setFWithTheseNodes(msh,face_node,cell_node,f,c,1,2); continue; }
    if (d==(MD_Negt|(MD_DirX+1)))
      { setFWithTheseNodes(msh,face_node,cell_node,f,c,0,3); continue; }
    if (d==(MD_Plus|(MD_DirY+1)))
      { setFWithTheseNodes(msh,face_node,cell_node,f,c,2,3); continue; }
    if (d==(MD_Negt|(MD_DirY+1)))
      { setFWithTheseNodes(msh,face_node,cell_node,f,c,0,1); continue; }
    fprintf(stderr,"[nabla_ini_face_node] Error!");
    exit(-1);
    //for(int n=0;n<NABLA_NODE_PER_CELL;n+=1)
    //  dbg(DBG_INI_FACE,"%%d ", cell_node[n*NABLA_NB_CELLS+c]);
  }
  if (DBG_DUMP) printf("\n[1;33mface->node:[0m");
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1){
    dbg(DBG_INI_FACE,"\n\tface #%%d: nodes ",f);
    if (DBG_DUMP) printf("\n");
    for(int n=0;n<msh.NABLA_NODE_PER_FACE;++n){
      dbg(DBG_INI_FACE,"%%d ",face_node[n*msh.NABLA_NB_FACES+f]);
      if (DBG_DUMP) printf(" %%d,",face_node[n*msh.NABLA_NB_FACES+f]);
      assert(face_node[n*msh.NABLA_NB_FACES+f]>=0);
    }
  }
}

// ****************************************************************************
// * xOf7 & yOf7
// ****************************************************************************
static double xOf7(const nablaMesh msh,
                   const int n){
  return
    ((double)(n%%msh.NABLA_NB_NODES_X_AXIS))*msh.NABLA_NB_NODES_X_TICK;
}
static double yOf7(const nablaMesh msh,
                   const int n){
  return
    ((double)((n/msh.NABLA_NB_NODES_X_AXIS)
              %%msh.NABLA_NB_NODES_Y_AXIS))*msh.NABLA_NB_NODES_Y_TICK;
}

// ****************************************************************************
// * Vérification des coordonnées
// ****************************************************************************
__attribute__((unused))
static void verifCoords(const nablaMesh msh,
                        Real3 *node_coord){
  dbg(DBG_INI_NODE,"\nVérification des coordonnés des noeuds");
  if (DBG_DUMP && DBG_LVL&DBG_INI_NODE) printf("\n[1;33mnode_coord:[0m");
  FOR_EACH_NODE_MSH(n){
    dbg(DBG_INI_NODE,"\n%%d:",n);
    if (DBG_DUMP && DBG_LVL&DBG_INI_NODE) printf("\n\t%%f,%%f,%%f,",node_coord[n].x,node_coord[n].y,node_coord[n].z);
    dbgReal3(DBG_INI_NODE,node_coord[n]);
  }
}

// ****************************************************************************
// * Initialisation des coordonnées
// ****************************************************************************
static void nabla_ini_node_coord(const nablaMesh msh,
                                 Real3 *node_coord){
  dbg(DBG_INI_NODE,"\nasserting NABLA_NB_NODES_Y_AXIS >= 1...");
  assert((msh.NABLA_NB_NODES_Y_AXIS >= 1));

  dbg(DBG_INI_NODE,"\nasserting (NABLA_NB_CELLS %% 1)==0...");
  assert((msh.NABLA_NB_CELLS %% 1)==0);
    
  for(int iNode=0; iNode<msh.NABLA_NB_NODES; iNode+=1){
    const int n=iNode;
    Real x,y;
    x=set(xOf7(msh,n));
    y=set(yOf7(msh,n));
    node_coord[iNode]=Real3(x,y,0.0);
  }
  verifCoords(msh,node_coord);
}

// ****************************************************************************
// * nabla_ini_connectivity
// ****************************************************************************
static void nabla_ini_connectivity(const nablaMesh msh,
                                   Real3 *node_coord,
                                   int *cell_node,
                                   int *cell_prev, int *cell_next,
                                   int* cell_face,
                                   int *node_cell,
                                   int* node_cell_corner,
                                   int* node_cell_and_corner,
                                   int* face_cell,
                                   int* face_node){
  nabla_ini_node_coord(msh,node_coord);
  nabla_ini_cell_node(msh,cell_node);
  nabla_ini_cell_next_prev(msh,cell_prev,cell_next);
  nabla_ini_node_cell(msh,
                      cell_node,
                      node_cell,
                      node_cell_corner,
                      node_cell_and_corner);
  nabla_ini_face_cell(msh,face_cell);
  nabla_ini_face_node(msh,face_cell,face_node,cell_node);
  nabla_ini_shift_back_face_cell(msh,face_cell);
  nabla_ini_cell_face(msh,face_cell,cell_node,face_node,cell_face);
  if (DBG_DUMP){
    printf("\n");
    exit(-1);
  }
  dbg(DBG_INI,"\nIni done");
}
