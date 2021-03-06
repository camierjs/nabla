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
// * 3D
// ****************************************************************************

// ****************************************************************************
// * Connectivité cell->node
// ****************************************************************************
__host__ static void nabla_ini_cell_node(const nablaMesh msh,
                                           int *cell_node){
  dbg(DBG_INI,"\nOn associe a chaque maille ses noeuds");
  int iCell=0;
  for(int iZ=0;iZ<msh.NABLA_NB_CELLS_Z_AXIS;iZ++){
    for(int iY=0;iY<msh.NABLA_NB_CELLS_Y_AXIS;iY++){
      for(int iX=0;iX<msh.NABLA_NB_CELLS_X_AXIS;iX++,iCell+=1){
        const int cell_uid=iX +
          iY*msh.NABLA_NB_CELLS_X_AXIS +
          iZ*msh.NABLA_NB_CELLS_X_AXIS*msh.NABLA_NB_CELLS_Y_AXIS;
        const int node_bid=iX +
          iY*msh.NABLA_NB_NODES_X_AXIS +
          iZ*msh.NABLA_NB_NODES_X_AXIS*msh.NABLA_NB_NODES_Y_AXIS;
        dbg(DBG_INI,"\n\tSetting cell #%%d %%dx%%dx%%d, cell_uid=%%d, node_bid=%%d",
            iCell,iX,iY,iZ,cell_uid,node_bid);
        cell_node[0*msh.NABLA_NB_CELLS+iCell] = node_bid;
        cell_node[1*msh.NABLA_NB_CELLS+iCell] = node_bid + 1;
        cell_node[2*msh.NABLA_NB_CELLS+iCell] = node_bid + msh.NABLA_NB_NODES_X_AXIS + 1;
        cell_node[3*msh.NABLA_NB_CELLS+iCell] = node_bid + msh.NABLA_NB_NODES_X_AXIS + 0;
        cell_node[4*msh.NABLA_NB_CELLS+iCell] = node_bid + msh.NABLA_NB_NODES_X_AXIS*msh.NABLA_NB_NODES_Y_AXIS;
        cell_node[5*msh.NABLA_NB_CELLS+iCell] = node_bid + msh.NABLA_NB_NODES_X_AXIS*msh.NABLA_NB_NODES_Y_AXIS +1 ;
        cell_node[6*msh.NABLA_NB_CELLS+iCell] = node_bid + msh.NABLA_NB_NODES_X_AXIS*msh.NABLA_NB_NODES_Y_AXIS + msh.NABLA_NB_NODES_X_AXIS+1;
        cell_node[7*msh.NABLA_NB_CELLS+iCell] = node_bid + msh.NABLA_NB_NODES_X_AXIS*msh.NABLA_NB_NODES_Y_AXIS + msh.NABLA_NB_NODES_X_AXIS;
        dbg(DBG_INI,"\n\tCell_%%d's nodes are %%d,%%d,%%d,%%d,%%d,%%d,%%d,%%d", iCell,
            cell_node[0*msh.NABLA_NB_CELLS+iCell],
            cell_node[1*msh.NABLA_NB_CELLS+iCell],
            cell_node[2*msh.NABLA_NB_CELLS+iCell],
            cell_node[3*msh.NABLA_NB_CELLS+iCell],
            cell_node[4*msh.NABLA_NB_CELLS+iCell],
            cell_node[5*msh.NABLA_NB_CELLS+iCell],
            cell_node[6*msh.NABLA_NB_CELLS+iCell],
            cell_node[7*msh.NABLA_NB_CELLS+iCell]);
      }
    }
  }
}
 
// ****************************************************************************
// * Vérification: Connectivité cell->next et cell->prev
// ****************************************************************************
__host__ __attribute__((unused)) static
void verifNextPrev(const nablaMesh msh,int *cell_prev, int *cell_next){
 for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI,"\nNext/Prev(X) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirX*msh.NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirX*msh.NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI,"\nNext/Prev(Y) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirY*msh.NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirY*msh.NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI,"\nNext/Prev(Z) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirZ*msh.NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirZ*msh.NABLA_NB_CELLS+i]);
  }
}

// ****************************************************************************
// * Connectivité cell->next et cell->prev
// ****************************************************************************
__host__ static void nabla_ini_cell_next_prev(const nablaMesh msh,
                                     int *cell_prev, int *cell_next){
  dbg(DBG_INI,"\nOn associe a chaque maille ses next et prev");
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
  // Dans la direction Z
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirZ*msh.NABLA_NB_CELLS+i] = i-msh.NABLA_NB_CELLS_X_AXIS*msh.NABLA_NB_CELLS_Y_AXIS ;
    cell_next[MD_DirZ*msh.NABLA_NB_CELLS+i] = i+msh.NABLA_NB_CELLS_X_AXIS*msh.NABLA_NB_CELLS_Y_AXIS ;
  }
  for (int i=0; i<msh.NABLA_NB_CELLS; ++i) {
    if (i<(msh.NABLA_NB_CELLS_X_AXIS*msh.NABLA_NB_CELLS_Y_AXIS)){
      cell_prev[MD_DirZ*msh.NABLA_NB_CELLS+i] = -77777777 ;
      cell_next[MD_DirZ*msh.NABLA_NB_CELLS+i+
                (msh.NABLA_NB_CELLS_X_AXIS*msh.NABLA_NB_CELLS_Y_AXIS)*(msh.NABLA_NB_CELLS_Z_AXIS-1)] = -88888888 ;
    }
  }
  verifNextPrev(msh,cell_prev,cell_next); 
}


// ****************************************************************************
// * qsort compare fonction for a Node and a Cell
// ****************************************************************************
__host__ static int comparNodeCell(const void *a, const void *b){
  return (*(int*)a)>(*(int*)b);
}
// ****************************************************************************
// * qsort compare fonction for a Node, Cell and Corner
// ****************************************************************************
__host__ static int comparNodeCellAndCorner(const void *pa, const void *pb){
  int *a=(int*)pa;
  int *b=(int*)pb;
  return a[0]>b[0];
}
// ****************************************************************************
// * Vérification: Connectivité node->cell et node->corner
// ****************************************************************************
__host__ __attribute__((unused)) static
void verifConnectivity(const nablaMesh msh,
                       int* node_cell,
                       int *node_cell_and_corner){
  dbg(DBG_INI,"\nVérification des connectivité des noeuds");
  FOR_EACH_NODE_MSH(n){
    dbg(DBG_INI,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL_MSH(c){
      dbg(DBG_INI,"\n\tnode_%%d knows cell %%d",n,node_cell[nc]);
      dbg(DBG_INI,", and node_%%d knows cell %%d",n,node_cell_and_corner[2*nc+0]);
    }
  }
}
__host__ __attribute__((unused)) static
void verifCorners(const nablaMesh msh,
                  int* node_cell,
                  int *node_cell_corner){
  dbg(DBG_INI,"\nVérification des coins des noeuds");
  FOR_EACH_NODE_MSH(n){
    dbg(DBG_INI,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL_MSH(c){
      if (node_cell_corner[nc]==-1) continue;
      dbg(DBG_INI,"\n\tnode_%%d is corner #%%d of cell %%d",n,
          node_cell_corner[nc],node_cell[nc]);
      //dbg(DBG_INI,", and node_%%d is corner #%%d of cell %%d",n,node_cell_and_corner[2*nc+1],node_cell_and_corner[2*nc+0]);
    }
  }
}

// ****************************************************************************
// * Connectivité node->cell et node->corner
// ****************************************************************************
__host__ static void nabla_ini_node_cell(const nablaMesh msh,
                                           const int* cell_node,
                                           int *node_cell,
                                           int* node_cell_corner,
                                           int* node_cell_and_corner){
  dbg(DBG_INI,"\nMaintenant, on re-scan pour remplir la connectivité des noeuds et des coins");
  dbg(DBG_INI,"\nOn flush le nombre de mailles attachées à ce noeud");
  // Padding is used to be sure we'll init all connectivities
  for(int n=0;n<msh.NABLA_NB_NODES+msh.NABLA_NODES_PADDING;n+=1){
    for(int c=0;c<8;++c){
      node_cell[8*n+c]=-1;
      node_cell_corner[8*n+c]=-1;
      node_cell_and_corner[2*(8*n+c)+0]=-1;//cell
      node_cell_and_corner[2*(8*n+c)+1]=-1;//corner
    }
  }  
  for(int c=0;c<msh.NABLA_NB_CELLS;c+=1){
    dbg(DBG_INI,"\nFocusing on cell %%d",c);
    for(int n=0;n<8;n++){
      const int iNode = cell_node[n*msh.NABLA_NB_CELLS+c];
      dbg(DBG_INI,"\n\tcell_%%d @%%d: pushs node %%d",c,n,iNode);
      // les 8 emplacements donnent l'offset jusqu'aux mailles
      // node_corner a une structure en 8*NABLA_NB_NODES
      node_cell[8*iNode+n]=c;
      node_cell_corner[8*iNode+n]=n;
      node_cell_and_corner[2*(8*iNode+n)+0]=c;//cell
      node_cell_and_corner[2*(8*iNode+n)+1]=n;//corner
    }
  }
  // On va maintenant trier les connectivités node->cell pour assurer l'associativité
  // void qsort(void *base, size_t nmemb, size_t size,
  //            int (*compar)(const void *, const void *));
  for(int n=0;n<msh.NABLA_NB_NODES;n+=1){
    qsort(&node_cell[8*n],8,sizeof(int),comparNodeCell);
    qsort(&node_cell_and_corner[2*8*n],8,2*sizeof(int),comparNodeCellAndCorner);
  }
  // And we come back to set our node_cell_corner
  for(int n=0;n<msh.NABLA_NB_NODES;n+=1)
    for(int c=0;c<8;++c)
      node_cell_corner[8*n+c]=node_cell_and_corner[2*(8*n+c)+1];
  //verifConnectivity();
  verifCorners(msh,node_cell,node_cell_corner);
}

// ****************************************************************************
// * Connectivité face->cell
// * Un encodage des diresction est utilisé en poids faibles:
// * bit:   2 1 0
// *     sign|.|.
// * MD_DirX => 1, MD_DirY => 2, MD_DirZ => 3
// ****************************************************************************
__host__ __attribute__((unused)) static int dir2bit(int d){
  assert((d==MD_DirX)||(d==MD_DirY)||(d==MD_DirZ));
  return d+1;
}
__host__ static char cXYZ(int d){
  assert((d==MD_DirX)||(d==MD_DirY)||(d==MD_DirZ));
  return (d==MD_DirX)?'X':(d==MD_DirY)?'Y':(d==MD_DirZ)?'Z':'?';
}
__host__ static char* sXYZ(int f){
  char str[32];
  const char sign=((f&4)==4)?'-':'+';
  f&=3;
  const char XorYorZ=(f==1)?'X':(f==2)?'Y':(f==3)?'Z':'?';
  assert(XorYorZ!='?');
  snprintf(str,32,"%%c%%c",sign,XorYorZ);
  return strdup(str);
}
__host__ static char* f2d(int f,bool shift=true){
  char str[32];
  if (f>=0 && shift) snprintf(str,32,"%%d",f>>MD_Shift);
  if (f>=0 && !shift) snprintf(str,32,"%%d",f);
  if (f<0) snprintf(str,32,"[1;31m%%s[m",sXYZ(-f));
  //if (f<0) snprintf(str,32,"%%s",sXYZ(-f));
  return strdup(str);
}
__host__ static int nabla_ini_face_cell_outer_minus(const nablaMesh msh,
                                                      int* face_cell,
                                                      const int *iof, const int c,
                                                      const int i, const int MD_Dir){
  const int f=iof[1];
  face_cell[0*msh.NABLA_NB_FACES+f] = (c<<MD_Shift)|MD_Negt|(MD_Dir+1);
  face_cell[1*msh.NABLA_NB_FACES+f] = -(MD_Negt|(MD_Dir+1));
  dbg(DBG_INI," %%s-%%c->%%s",
      f2d(face_cell[0*msh.NABLA_NB_FACES+f]),
      cXYZ(MD_Dir),
      f2d(face_cell[1*msh.NABLA_NB_FACES+f]));
  return 1;
}
__host__ static int nabla_ini_face_cell_inner(const nablaMesh msh,
                                                int* face_cell,
                                                const int *iof, const int c,
                                                const int i, const int MD_Dir){
  const int f=iof[0];
  face_cell[0*msh.NABLA_NB_FACES+f] = (c<<MD_Shift)|MD_Plus|(MD_Dir+1);
  if (MD_Dir==MD_DirX) face_cell[1*msh.NABLA_NB_FACES+f] = c+1;
  if (MD_Dir==MD_DirY) face_cell[1*msh.NABLA_NB_FACES+f] = c+msh.NABLA_NB_CELLS_X_AXIS;
  if (MD_Dir==MD_DirZ) face_cell[1*msh.NABLA_NB_FACES+f] = c+msh.NABLA_NB_CELLS_X_AXIS*msh.NABLA_NB_CELLS_Y_AXIS;
  face_cell[1*msh.NABLA_NB_FACES+f] <<= MD_Shift;
  face_cell[1*msh.NABLA_NB_FACES+f] |= (MD_Plus|(MD_Dir+1));
  dbg(DBG_INI," %%s-%%c->%%s",
      f2d(face_cell[0*msh.NABLA_NB_FACES+f]),
      cXYZ(MD_Dir),
      f2d(face_cell[1*msh.NABLA_NB_FACES+f]));
  return 1;
}
__host__ static int nabla_ini_face_cell_outer_plus(const nablaMesh msh,
                                                     int* face_cell,
                                                     const int *iof, const int c,
                                                     const int i, const int MD_Dir){
  const int f=iof[1];
  face_cell[0*msh.NABLA_NB_FACES+f] = (c<<MD_Shift)|MD_Plus|(MD_Dir+1);
  face_cell[1*msh.NABLA_NB_FACES+f] = -(MD_Plus|(MD_Dir+1));
  dbg(DBG_INI," %%s-%%c->%%s",
      f2d(face_cell[0*msh.NABLA_NB_FACES+f]),
      cXYZ(MD_Dir),
      f2d(face_cell[1*msh.NABLA_NB_FACES+f]));
  return 1;
}
__host__ static void nabla_ini_face_cell_XYZ(const nablaMesh msh,
                                               int* face_cell,
                                               int *f, const int c,
                                               const int i, const int MD_Dir){
  const int n =
    (MD_Dir==MD_DirX)?msh.NABLA_NB_CELLS_X_AXIS:
    (MD_Dir==MD_DirY)?msh.NABLA_NB_CELLS_Y_AXIS:
    (MD_Dir==MD_DirZ)?msh.NABLA_NB_CELLS_Z_AXIS:-0xDEADBEEF;  
  if (i<n-1)  f[0]+=nabla_ini_face_cell_inner(msh,face_cell,f,c,i,MD_Dir);
  if (i==0)   f[1]+=nabla_ini_face_cell_outer_minus(msh,face_cell,f,c,i,MD_Dir);
  if (i==n-1) f[1]+=nabla_ini_face_cell_outer_plus(msh,face_cell,f,c,i,MD_Dir);
}
__host__ static void nabla_ini_face_cell(const nablaMesh msh,
                                           int* face_cell){
  dbg(DBG_INI,"\n[1;33mOn associe a chaque maille ses faces:[m");
  int f[2]={0,msh.NABLA_NB_FACES_INNER}; // inner and outer faces
  for(int iZ=0;iZ<msh.NABLA_NB_CELLS_Z_AXIS;iZ++){
    for(int iY=0;iY<msh.NABLA_NB_CELLS_Y_AXIS;iY++){
      for(int iX=0;iX<msh.NABLA_NB_CELLS_X_AXIS;iX++){
        const int c=iX + iY*msh.NABLA_NB_CELLS_X_AXIS + iZ*msh.NABLA_NB_CELLS_X_AXIS*msh.NABLA_NB_CELLS_Y_AXIS;
        dbg(DBG_INI,"\n\tCell #[1;36m%%d[m @ %%dx%%dx%%d:",c,iX,iY,iZ);
        nabla_ini_face_cell_XYZ(msh,face_cell,f,c,iX,MD_DirX);
        nabla_ini_face_cell_XYZ(msh,face_cell,f,c,iZ,MD_DirZ);
        nabla_ini_face_cell_XYZ(msh,face_cell,f,c,iY,MD_DirY);
      }
    }
  }
  dbg(DBG_INI,"\n\tNumber of faces = %%d",f[0]+f[1]-msh.NABLA_NB_FACES_INNER);
  assert(f[0]==msh.NABLA_NB_FACES_INNER);
  assert(f[1]==msh.NABLA_NB_FACES_INNER+msh.NABLA_NB_FACES_OUTER);
  assert((f[0]+f[1])==msh.NABLA_NB_FACES+msh.NABLA_NB_FACES_INNER);
  // On laisse les faces shiftées/encodées avec les directions pour les face_node
}

// ****************************************************************************
// * On les a shifté pour connaitre les directions, on flush les positifs
// ****************************************************************************
__host__ static void nabla_ini_shift_back_face_cell(const nablaMesh msh,
                                                      int* face_cell){
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1){
    if (face_cell[0*msh.NABLA_NB_FACES+f]>0) face_cell[0*msh.NABLA_NB_FACES+f]>>=MD_Shift;
    if (face_cell[1*msh.NABLA_NB_FACES+f]>0) face_cell[1*msh.NABLA_NB_FACES+f]>>=MD_Shift;
  }
  dbg(DBG_INI,"\n[nabla_ini_shift_back_face_cell] Inner faces:\n");
  for(int f=0;f<msh.NABLA_NB_FACES_INNER;f+=1)
    dbg(DBG_INI," %%s->%%s",
        f2d(face_cell[0*msh.NABLA_NB_FACES+f],false),
        f2d(face_cell[1*msh.NABLA_NB_FACES+f],false));
  dbg(DBG_INI,"\n[nabla_ini_shift_back_face_cell] Outer faces:\n");
  for(int f=msh.NABLA_NB_FACES_INNER;f<msh.NABLA_NB_FACES_INNER+msh.NABLA_NB_FACES_OUTER;f+=1)
    dbg(DBG_INI," %%s->%%s",
        f2d(face_cell[0*msh.NABLA_NB_FACES+f],false),
        f2d(face_cell[1*msh.NABLA_NB_FACES+f],false));
  dbg(DBG_INI,"\n[nabla_ini_shift_back_face_cell] All faces:\n");
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1)
    dbg(DBG_INI," %%d->%%d",
        face_cell[0*msh.NABLA_NB_FACES+f],
        face_cell[1*msh.NABLA_NB_FACES+f]);
}

// ****************************************************************************
// * Connectivité cell->face
// ****************************************************************************
__host__ static void addThisfaceToCellConnectivity(const nablaMesh msh,
                                          int* cell_face,
                                          const int f, const int c){
  dbg(DBG_INI,"\n\t\t[addThisfaceToCellConnectivity] Adding face #%%d to cell %%d ",f,c);
  for(int i=0;i<msh.NABLA_FACE_PER_CELL;i+=1){
    // On scrute le premier emplacement 
    if (cell_face[i*msh.NABLA_NB_CELLS+c]>=0) continue;
    dbg(DBG_INI,"[%%d] ",i);
    cell_face[i*msh.NABLA_NB_CELLS+c]=f;
    break; // We're finished here
  }
}
__host__ static void nabla_ini_cell_face(const nablaMesh msh,
                                const int* face_cell,
                                int* cell_face){
  dbg(DBG_INI,"\n[1;33mOn revient pour remplir cell->face:[m (flushing)");
  for(int c=0;c<msh.NABLA_NB_CELLS;c+=1){
    for(int f=0;f<msh.NABLA_FACE_PER_CELL;f+=1){
      cell_face[f*msh.NABLA_NB_CELLS+c]=-1;
    }
  }
 
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1){
    const int cell0 = face_cell[0*msh.NABLA_NB_FACES+f];
    const int cell1 = face_cell[1*msh.NABLA_NB_FACES+f];
    dbg(DBG_INI,"\n\t[nabla_ini_cell_face] Pushing face #%%d: %%d->%%d",f,cell0,cell1);
    if (cell0>=0) addThisfaceToCellConnectivity(msh,cell_face,f,cell0);
    if (cell1>=0) addThisfaceToCellConnectivity(msh,cell_face,f,cell1);
  }

  dbg(DBG_INI,"\n[1;33mOn revient pour dumper cell->face:[m");
  for(int c=0;c<msh.NABLA_NB_CELLS;c+=1){
    for(int f=0;f<msh.NABLA_FACE_PER_CELL;f+=1){
      if (cell_face[f*msh.NABLA_NB_CELLS+c]<0) continue;
      dbg(DBG_INI,"\n\t[nabla_ini_cell_face] cell[%%d]_face[%%d] %%d",
          c,f,cell_face[f*msh.NABLA_NB_CELLS+c]);
    }
  }
}


// ****************************************************************************
// * Connectivité face->node
// ****************************************************************************
__host__ static const char* i2XYZ(const int i){
  const int c=i&MD_Mask;
  //dbg(DBG_INI,"\n\t\t[33m[i2XYZ] i=%%d, Shift=%%d, c=%%d[m",i,MD_Shift,c);
  if (c==(MD_Plus|(MD_DirX+1))) return strdup("x+");
  if (c==(MD_Negt|(MD_DirX+1))) return strdup("x-");
  if (c==(MD_Plus|(MD_DirY+1))) return strdup("y+");
  if (c==(MD_Negt|(MD_DirY+1))) return strdup("y-");
  if (c==(MD_Plus|(MD_DirZ+1))) return strdup("z+");
  if (c==(MD_Negt|(MD_DirZ+1))) return strdup("z-");
  fprintf(stderr,"[i2XYZ] Error, could not distinguish XYZ!");
  exit(-1);
  return NULL;
}
__host__ static const char* c2XYZ(const int c){
  char str[16];
  if (c==-(MD_Negt|(MD_DirX+1))) return strdup("[1;31mX-[m");
  if (c==-(MD_Plus|(MD_DirX+1))) return strdup("[1;31mX+[m");
  if (c==-(MD_Negt|(MD_DirY+1))) return strdup("[1;31mY-[m");
  if (c==-(MD_Plus|(MD_DirY+1))) return strdup("[1;31mY+[m");
  if (c==-(MD_Negt|(MD_DirZ+1))) return strdup("[1;31mZ-[m");
  if (c==-(MD_Plus|(MD_DirZ+1))) return strdup("[1;31mZ+[m");
  if (snprintf(str,16,"%%d%%s",c>>MD_Shift,i2XYZ(c))<0) fprintf(stderr,"c2XYZ!");
  return strdup(str);
}
__host__ static void setFWithTheseNodes(const nablaMesh msh,
                                          int* face_node,
                                          int* cell_node,
                                          const int f, const int c,
                                          const int n0, const int n1,
                                          const int n2, const int n3){
  face_node[0*msh.NABLA_NB_FACES+f]=cell_node[n0*msh.NABLA_NB_CELLS+c];
  face_node[1*msh.NABLA_NB_FACES+f]=cell_node[n1*msh.NABLA_NB_CELLS+c];
  face_node[2*msh.NABLA_NB_FACES+f]=cell_node[n2*msh.NABLA_NB_CELLS+c];
  face_node[3*msh.NABLA_NB_FACES+f]=cell_node[n3*msh.NABLA_NB_CELLS+c];
}
__host__ static void nabla_ini_face_node(const nablaMesh msh,
                                           const int* face_cell,
                                           int* face_node,
                                           int* cell_node){
  dbg(DBG_INI,"\n[1;33mOn associe a chaque faces ses noeuds:[m");
  // On flush toutes les connectivités face_noeuds
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1)
    for(int n=0;n<msh.NABLA_NODE_PER_FACE;++n)
      face_node[n*msh.NABLA_NB_FACES+f]=-1;
  
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1){
    const int backCell=face_cell[0*msh.NABLA_NB_FACES+f];
    const int frontCell=face_cell[1*msh.NABLA_NB_FACES+f];
    dbg(DBG_INI,"\n\tFace #[1;36m%%d[m: %%d => %%d, ",f, backCell, frontCell);
    dbg(DBG_INI,"\t%%s => %%s: ", c2XYZ(backCell), c2XYZ(frontCell));
    // On va travailler avec sa backCell
    const int c=backCell>>MD_Shift;
    const int d=backCell &MD_Mask;
    dbg(DBG_INI,"\t%%d ", c);
    assert(c>=0);
    if (d==(MD_Plus|(MD_DirX+1))) { setFWithTheseNodes(msh,face_node,cell_node,f,c,1,2,5,6); continue; }
    if (d==(MD_Negt|(MD_DirX+1))) { setFWithTheseNodes(msh,face_node,cell_node,f,c,0,3,4,7); continue; }
    if (d==(MD_Plus|(MD_DirY+1))) { setFWithTheseNodes(msh,face_node,cell_node,f,c,2,3,6,7); continue; }
    if (d==(MD_Negt|(MD_DirY+1))) { setFWithTheseNodes(msh,face_node,cell_node,f,c,0,1,4,5); continue; }
    if (d==(MD_Plus|(MD_DirZ+1))) { setFWithTheseNodes(msh,face_node,cell_node,f,c,4,5,6,7); continue; }
    if (d==(MD_Negt|(MD_DirZ+1))) { setFWithTheseNodes(msh,face_node,cell_node,f,c,0,1,2,3); continue; }
    fprintf(stderr,"[nabla_ini_face_node] Error!");
    exit(-1);
    //for(int n=0;n<NABLA_NODE_PER_CELL;n+=1)
    //  dbg(DBG_INI,"%%d ", cell_node[n*NABLA_NB_CELLS+c]);
  }
  for(int f=0;f<msh.NABLA_NB_FACES;f+=1)
    for(int n=0;n<msh.NABLA_NODE_PER_FACE;++n)
      assert(face_node[n*msh.NABLA_NB_FACES+f]>=0);  
}

// ****************************************************************************
// * xOf7 & yOf7
// ****************************************************************************
__host__ static double xOf7(const nablaMesh msh,const int n){
  return
    ((double)(n%%msh.NABLA_NB_NODES_X_AXIS))*msh.NABLA_NB_NODES_X_TICK;
}
__host__ static double yOf7(const nablaMesh msh,const int n){
  return
    ((double)((n/msh.NABLA_NB_NODES_X_AXIS)
              %%msh.NABLA_NB_NODES_Y_AXIS))*msh.NABLA_NB_NODES_Y_TICK;
}
__host__ static double zOf7(const nablaMesh msh,const int n){
  return
    ((double)((n/(msh.NABLA_NB_NODES_X_AXIS*msh.NABLA_NB_NODES_Y_AXIS))
              %%msh.NABLA_NB_NODES_Z_AXIS))*msh.NABLA_NB_NODES_Z_TICK;
}

// ****************************************************************************
// * Vérification des coordonnées
// ****************************************************************************
__host__ __attribute__((unused)) static
void verifCoords(const nablaMesh msh,Real3 *node_coord){
  dbg(DBG_INI,"\nVérification des coordonnés des noeuds");
  FOR_EACH_NODE_MSH(n){
    dbg(DBG_INI,"\n%%d:",n);
    dbgReal3(DBG_INI,node_coord[n]);
  }
}

// ****************************************************************************
// * Initialisation des coordonnées
// ****************************************************************************
__host__ static void nabla_ini_node_coord(const nablaMesh msh,
                                          Real3 *node_coord){
  dbg(DBG_INI,"\nasserting NABLA_NB_NODES_Y_AXIS >= 1...");
  assert((msh.NABLA_NB_NODES_Y_AXIS >= WARP_SIZE));

  dbg(DBG_INI,"\nasserting (NABLA_NB_CELLS %% 1)==0...");
  assert((msh.NABLA_NB_CELLS %% WARP_SIZE)==0);
  
  for(int iNode=0; iNode<msh.NABLA_NB_NODES_WARP; iNode+=1){
    const int n=WARP_SIZE*iNode;
    Real x,y,z;
#if defined(__MIC__)||defined(__AVX512__)
    x=set(xOf7(msh,n+7), xOf7(msh,n+6), xOf7(msh,n+5), xOf7(msh,n+4), xOf7(msh,n+3), xOf7(msh,n+2), xOf7(msh,n+1), xOf7(msh,n));
    y=set(yOf7(msh,n+7), yOf7(msh,n+6), yOf7(msh,n+5), yOf7(msh,n+4), yOf7(msh,n+3), yOf7(msh,n+2), yOf7(msh,n+1), yOf7(msh,n));
    z=set(zOf7(msh,n+7), zOf7(msh,n+6), zOf7(msh,n+5), zOf7(msh,n+4), zOf7(msh,n+3), zOf7(msh,n+2), zOf7(msh,n+1), zOf7(msh,n));
#elif (__AVX__ || __AVX2__) && !defined(NO_SSE2)
    x=set(xOf7(msh,n+3), xOf7(msh,n+2), xOf7(msh,n+1), xOf7(msh,n));
    y=set(yOf7(msh,n+3), yOf7(msh,n+2), yOf7(msh,n+1), yOf7(msh,n));
    z=set(zOf7(msh,n+3), zOf7(msh,n+2), zOf7(msh,n+1), zOf7(msh,n));
#elif __SSE2__ && !defined(NO_SSE2)
    x=set(xOf7(msh,n+1), xOf7(msh,n));
    y=set(yOf7(msh,n+1), yOf7(msh,n));
    z=set(zOf7(msh,n+1), zOf7(msh,n));
#else
    x=set(xOf7(msh,n));
    y=set(yOf7(msh,n));
    z=set(zOf7(msh,n));
#endif
    node_coord[iNode]=Real3(x,y,z);
    //dbgReal3(DBG_INI,node_coord[iNode]);
  }
  verifCoords(msh,node_coord);
}

// ****************************************************************************
// * nabla_ini_connectivity
// ****************************************************************************
__host__ static void nabla_ini_connectivity(const nablaMesh msh,
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
  nabla_ini_node_cell(msh,cell_node,
                      node_cell,
                      node_cell_corner,
                      node_cell_and_corner);
  nabla_ini_face_cell(msh,face_cell);
  nabla_ini_face_node(msh,face_cell,face_node,cell_node);
  nabla_ini_shift_back_face_cell(msh,face_cell);
  nabla_ini_cell_face(msh,face_cell,cell_face);
  dbg(DBG_INI,"\nIni done");
}
