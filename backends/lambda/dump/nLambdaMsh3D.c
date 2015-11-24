///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
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
// * Forward declarations
// ****************************************************************************
static void verifCoords(void);
static void verifCorners(void);
static void verifNextPrev(void);
static void verifConnectivity(void);
static int comparNodeCell(const void*, const void*);
static int comparNodeCellAndCorner(const void*, const void*);
static double xOf7(const int);
static double yOf7(const int);
static double zOf7(const int);
//static int lg2(int p){int i=0; for(;p!=(0ull);++i,p>>=(1ull)); return i;}


// ****************************************************************************
// * Connectivité cell->node
// ****************************************************************************
static void nabla_ini_cell_node(void){
  dbg(DBG_OFF,"\nOn associe a chaque maille ses noeuds");
  int iCell=0;
  for(int iZ=0;iZ<NABLA_NB_CELLS_Z_AXIS;iZ++){
    for(int iY=0;iY<NABLA_NB_CELLS_Y_AXIS;iY++){
      for(int iX=0;iX<NABLA_NB_CELLS_X_AXIS;iX++,iCell+=1){
        const int cell_uid=iX + iY*NABLA_NB_CELLS_X_AXIS + iZ*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS;
        const int node_bid=iX + iY*NABLA_NB_NODES_X_AXIS + iZ*NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;
        dbg(DBG_OFF,"\n\tSetting cell #%%d %%dx%%dx%%d, cell_uid=%%d, node_bid=%%d",
            iCell,iX,iY,iZ,cell_uid,node_bid);
        cell_node[0*NABLA_NB_CELLS+iCell] = node_bid;
        cell_node[1*NABLA_NB_CELLS+iCell] = node_bid + 1;
        cell_node[2*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS + 1;
        cell_node[3*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS + 0;
        cell_node[4*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;
        cell_node[5*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS +1 ;
        cell_node[6*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS + NABLA_NB_NODES_X_AXIS+1;
        cell_node[7*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS + NABLA_NB_NODES_X_AXIS;
        dbg(DBG_OFF,"\n\tCell_%%d's nodes are %%d,%%d,%%d,%%d,%%d,%%d,%%d,%%d", iCell,
            cell_node[0*NABLA_NB_CELLS+iCell],
            cell_node[1*NABLA_NB_CELLS+iCell],
            cell_node[2*NABLA_NB_CELLS+iCell],
            cell_node[3*NABLA_NB_CELLS+iCell],
            cell_node[4*NABLA_NB_CELLS+iCell],
            cell_node[5*NABLA_NB_CELLS+iCell],
            cell_node[6*NABLA_NB_CELLS+iCell],
            cell_node[7*NABLA_NB_CELLS+iCell]);
      }
    }
  }
}

 
// ****************************************************************************
// * Connectivité cell->next et cell->prev
// ****************************************************************************
static void nabla_ini_cell_next_prev(void){
 dbg(DBG_OFF,"\nOn associe a chaque maille ses next et prev");
  // On met des valeurs négatives afin que le gatherk_and_zero_neg_ones puisse les reconaitre
  // Dans la direction X
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirX*NABLA_NB_CELLS+i] = i-1 ;
    cell_next[MD_DirX*NABLA_NB_CELLS+i] = i+1 ;
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    if ((i%%NABLA_NB_CELLS_X_AXIS)==0){
      cell_prev[MD_DirX*NABLA_NB_CELLS+i] = -33333333 ;
      cell_next[MD_DirX*NABLA_NB_CELLS+i+NABLA_NB_CELLS_X_AXIS-1] = -44444444 ;
    }
  }
  // Dans la direction Y
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirY*NABLA_NB_CELLS+i] = i-NABLA_NB_CELLS_X_AXIS ;
    cell_next[MD_DirY*NABLA_NB_CELLS+i] = i+NABLA_NB_CELLS_X_AXIS ;
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    if ((i%%(NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS))<NABLA_NB_CELLS_Y_AXIS){
      cell_prev[MD_DirY*NABLA_NB_CELLS+i] = -55555555 ;
      cell_next[MD_DirY*NABLA_NB_CELLS+i+
                (NABLA_NB_CELLS_X_AXIS-1)*NABLA_NB_CELLS_Y_AXIS] = -66666666 ;
    }
  }
  // Dans la direction Z
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirZ*NABLA_NB_CELLS+i] = i-NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS ;
    cell_next[MD_DirZ*NABLA_NB_CELLS+i] = i+NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS ;
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    if (i<(NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS)){
      cell_prev[MD_DirZ*NABLA_NB_CELLS+i] = -77777777 ;
      cell_next[MD_DirZ*NABLA_NB_CELLS+i+
                (NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS)*(NABLA_NB_CELLS_Z_AXIS-1)] = -88888888 ;
    }
  }
  verifNextPrev(); 
}
// ****************************************************************************
// * Vérification: Connectivité cell->next et cell->prev
// ****************************************************************************
__attribute__((unused)) static void verifNextPrev(void){
 for (int i=0; i<NABLA_NB_CELLS; ++i) {
    dbg(DBG_OFF,"\nNext/Prev(X) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirX*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirX*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    dbg(DBG_OFF,"\nNext/Prev(Y) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirY*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirY*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    dbg(DBG_OFF,"\nNext/Prev(Z) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirZ*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirZ*NABLA_NB_CELLS+i]);
  }
}


// ****************************************************************************
// * Connectivité node->cell et node->corner
// ****************************************************************************
static void nabla_ini_node_cell(void){
  dbg(DBG_OFF,"\nMaintenant, on re-scan pour remplir la connectivité des noeuds et des coins");
  dbg(DBG_OFF,"\nOn flush le nombre de mailles attachées à ce noeud");
  for(int n=0;n<NABLA_NB_NODES;n+=1){
    for(int c=0;c<8;++c){
      node_cell[8*n+c]=-1;
      node_cell_corner[8*n+c]=-1;
      node_cell_and_corner[2*(8*n+c)+0]=-1;//cell
      node_cell_and_corner[2*(8*n+c)+1]=-1;//corner
    }
  }  
  for(int c=0;c<NABLA_NB_CELLS;c+=1){
    dbg(DBG_OFF,"\nFocusing on cell %%d",c);
    for(int n=0;n<8;n++){
      const int iNode = cell_node[n*NABLA_NB_CELLS+c];
      dbg(DBG_OFF,"\n\tcell_%%d @%%d: pushs node %%d",c,n,iNode);
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
  for(int n=0;n<NABLA_NB_NODES;n+=1){
    qsort(&node_cell[8*n],8,sizeof(int),comparNodeCell);
    qsort(&node_cell_and_corner[2*8*n],8,2*sizeof(int),comparNodeCellAndCorner);
  }
  // And we come back to set our node_cell_corner
  for(int n=0;n<NABLA_NB_NODES;n+=1)
    for(int c=0;c<8;++c)
      node_cell_corner[8*n+c]=node_cell_and_corner[2*(8*n+c)+1];
  //verifConnectivity();
  //verifCorners();
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
__attribute__((unused)) static void verifConnectivity(void){
  dbg(DBG_OFF,"\nVérification des connectivité des noeuds");
  FOR_EACH_NODE(n){
    dbg(DBG_OFF,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL(c){
      dbg(DBG_OFF,"\n\tnode_%%d knows cell %%d",n,node_cell[nc]);
      dbg(DBG_OFF,", and node_%%d knows cell %%d",n,node_cell_and_corner[2*nc+0]);
    }
  }
}
__attribute__((unused)) static void verifCorners(void){
  dbg(DBG_OFF,"\nVérification des coins des noeuds");
  FOR_EACH_NODE(n){
    dbg(DBG_OFF,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL(c){
      if (node_cell_corner[nc]==-1) continue;
      dbg(DBG_OFF,"\n\tnode_%%d is corner #%%d of cell %%d",n,node_cell_corner[nc],node_cell[nc]);
      //dbg(DBG_OFF,", and node_%%d is corner #%%d of cell %%d",n,node_cell_and_corner[2*nc+1],node_cell_and_corner[2*nc+0]);
    }
  }
}


// ****************************************************************************
// * Connectivité face->cell
// * Un encodage des diresction est utilisé en poids faibles:
// * bit:   2 1 0
// *     sign|.|.
// * MD_DirX => 1, MD_DirY => 2, MD_DirZ => 3
// ****************************************************************************
__attribute__((unused)) static const int dir2bit(int d){
  assert((d==MD_DirX)||(d==MD_DirY)||(d==MD_DirZ));
  return d+1;
}
static const char cXYZ(int d){
  assert((d==MD_DirX)||(d==MD_DirY)||(d==MD_DirZ));
  return (d==MD_DirX)?'X':(d==MD_DirY)?'Y':(d==MD_DirZ)?'Z':'?';
}
static const char* sXYZ(int f){
  char str[32];
  const char sign=((f&4)==4)?'-':'+';
  f&=3;
  const char XorYorZ=(f==1)?'X':(f==2)?'Y':(f==3)?'Z':'?';
  assert(XorYorZ!='?');
  snprintf(str,32,"%%c%%c",sign,XorYorZ);
  return strdup(str);
}
static char* f2d(int f,bool shift=true){
  char str[32];
  if (f>=0 && shift) snprintf(str,32,"%%d",f>>MD_Shift);
  if (f>=0 && !shift) snprintf(str,32,"%%d",f);
  if (f<0) snprintf(str,32,"[1;31m%%s[m",sXYZ(-f));
  //if (f<0) snprintf(str,32,"%%s",sXYZ(-f));
  return strdup(str);
}
static int nabla_ini_face_cell_outer_minus(const int *iof, const int c,
                                           const int i, const int MD_Dir){
  const int f=iof[1];
  face_cell[0*NABLA_NB_FACES+f] = (c<<MD_Shift)|MD_Negt|(MD_Dir+1);
  face_cell[1*NABLA_NB_FACES+f] = -(MD_Negt|(MD_Dir+1));
  dbg(DBG_OFF," %%s-%%c->%%s",
      f2d(face_cell[0*NABLA_NB_FACES+f]),
      cXYZ(MD_Dir),
      f2d(face_cell[1*NABLA_NB_FACES+f]));
  return 1;
}
static int nabla_ini_face_cell_inner(const int *iof, const int c,
                                     const int i, const int MD_Dir){
  const int f=iof[0];
  face_cell[0*NABLA_NB_FACES+f] = (c<<MD_Shift)|MD_Plus|(MD_Dir+1);
  if (MD_Dir==MD_DirX) face_cell[1*NABLA_NB_FACES+f] = c+1;
  if (MD_Dir==MD_DirY) face_cell[1*NABLA_NB_FACES+f] = c+NABLA_NB_CELLS_X_AXIS;
  if (MD_Dir==MD_DirZ) face_cell[1*NABLA_NB_FACES+f] = c+NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS;
  face_cell[1*NABLA_NB_FACES+f] <<= MD_Shift;
  face_cell[1*NABLA_NB_FACES+f] |= (MD_Plus|(MD_Dir+1));
  dbg(DBG_OFF," %%s-%%c->%%s",
      f2d(face_cell[0*NABLA_NB_FACES+f]),
      cXYZ(MD_Dir),
      f2d(face_cell[1*NABLA_NB_FACES+f]));
  return 1;
}
static int nabla_ini_face_cell_outer_plus(const int *iof, const int c,
                                          const int i, const int MD_Dir){
  const int f=iof[1];
  face_cell[0*NABLA_NB_FACES+f] = (c<<MD_Shift)|MD_Plus|(MD_Dir+1);
  face_cell[1*NABLA_NB_FACES+f] = -(MD_Plus|(MD_Dir+1));
  dbg(DBG_OFF," %%s-%%c->%%s",
      f2d(face_cell[0*NABLA_NB_FACES+f]),
      cXYZ(MD_Dir),
      f2d(face_cell[1*NABLA_NB_FACES+f]));
  return 1;
}
static void nabla_ini_face_cell_XYZ(int *f, const int c,
                                   const int i, const int MD_Dir){
  const int n =
    (MD_Dir==MD_DirX)?NABLA_NB_CELLS_X_AXIS:
    (MD_Dir==MD_DirY)?NABLA_NB_CELLS_Y_AXIS:
    (MD_Dir==MD_DirZ)?NABLA_NB_CELLS_Z_AXIS:-0xDEADBEEFull;  
  if (i<n-1)  f[0]+=nabla_ini_face_cell_inner(f,c,i,MD_Dir);
  if (i==0)   f[1]+=nabla_ini_face_cell_outer_minus(f,c,i,MD_Dir);
  if (i==n-1) f[1]+=nabla_ini_face_cell_outer_plus(f,c,i,MD_Dir);
}
static void nabla_ini_face_cell(void){
  dbg(DBG_OFF,"\n[1;33mOn associe a chaque maille ses faces:[m");
  int f[2]={0,NABLA_NB_FACES_INNER}; // inner and outer faces
  for(int iZ=0;iZ<NABLA_NB_CELLS_Z_AXIS;iZ++){
    for(int iY=0;iY<NABLA_NB_CELLS_Y_AXIS;iY++){
      for(int iX=0;iX<NABLA_NB_CELLS_X_AXIS;iX++){
        const int c=iX + iY*NABLA_NB_CELLS_X_AXIS + iZ*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS;
        dbg(DBG_OFF,"\n\tCell #[1;36m%%d[m @ %%dx%%dx%%d:",c,iX,iY,iZ);
        nabla_ini_face_cell_XYZ(f,c,iX,MD_DirX);
        nabla_ini_face_cell_XYZ(f,c,iZ,MD_DirZ);
        nabla_ini_face_cell_XYZ(f,c,iY,MD_DirY);
      }
    }
  }
  dbg(DBG_OFF,"\n\tNumber of faces = %%d",f[0]+f[1]-NABLA_NB_FACES_INNER);
  assert(f[0]==NABLA_NB_FACES_INNER);
  assert(f[1]==NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER);
  assert((f[0]+f[1])==NABLA_NB_FACES+NABLA_NB_FACES_INNER);
  // On laisse les faces shiftées/encodées avec les directions pour les face_node
}
// ****************************************************************************
// * On les a shifté pour connaitre les directions, on flush les positifs
// ****************************************************************************
void nabla_ini_shift_back_face_cell(void){
  for(int f=0;f<NABLA_NB_FACES;f+=1){
    if (face_cell[0*NABLA_NB_FACES+f]>0) face_cell[0*NABLA_NB_FACES+f]>>=MD_Shift;
    if (face_cell[1*NABLA_NB_FACES+f]>0) face_cell[1*NABLA_NB_FACES+f]>>=MD_Shift;
  }
  dbg(DBG_OFF,"\n[nabla_ini_shift_back_face_cell] Inner faces:\n");
  for(int f=0;f<NABLA_NB_FACES_INNER;f+=1)
    dbg(DBG_OFF," %%s->%%s",
        f2d(face_cell[0*NABLA_NB_FACES+f],false),
        f2d(face_cell[1*NABLA_NB_FACES+f],false));
  dbg(DBG_OFF,"\n[nabla_ini_shift_back_face_cell] Outer faces:\n");
  for(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)
    dbg(DBG_OFF," %%s->%%s",
        f2d(face_cell[0*NABLA_NB_FACES+f],false),
        f2d(face_cell[1*NABLA_NB_FACES+f],false));
  dbg(DBG_OFF,"\n[nabla_ini_shift_back_face_cell] All faces:\n");
  for(int f=0;f<NABLA_NB_FACES;f+=1)
    dbg(DBG_OFF," %%d->%%d",
        face_cell[0*NABLA_NB_FACES+f],
        face_cell[1*NABLA_NB_FACES+f]);
}


// ****************************************************************************
// * Connectivité face->node
// ****************************************************************************
static const char* i2XYZ(const int i){
  const int c=i&MD_Mask;
  //dbg(DBG_OFF,"\n\t\t[33m[i2XYZ] i=%%d, Shift=%%d, c=%%d[m",i,MD_Shift,c);
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
static const char* c2XYZ(const int c){
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
static void setFWithTheseNodes(const int f, const int c,
                               const int n0, const int n1,
                               const int n2, const int n3){
  face_node[0*NABLA_NB_FACES+f]=cell_node[n0*NABLA_NB_CELLS+c];
  face_node[1*NABLA_NB_FACES+f]=cell_node[n1*NABLA_NB_CELLS+c];
  face_node[2*NABLA_NB_FACES+f]=cell_node[n2*NABLA_NB_CELLS+c];
  face_node[3*NABLA_NB_FACES+f]=cell_node[n3*NABLA_NB_CELLS+c];
}
static void nabla_ini_face_node(void){
  dbg(DBG_OFF,"\n[1;33mOn associe a chaque faces ses noeuds:[m");
  // On flush toutes les connectivités face_noeuds
  for(int f=0;f<NABLA_NB_FACES;f+=1)
    for(int n=0;n<NABLA_NODE_PER_FACE;++n)
      face_node[n*NABLA_NB_FACES+f]=-1;
  
  for(int f=0;f<NABLA_NB_FACES;f+=1){
    const int backCell=face_cell[0*NABLA_NB_FACES+f];
    const int frontCell=face_cell[1*NABLA_NB_FACES+f];
    dbg(DBG_OFF,"\n\tFace #[1;36m%%d[m: %%d => %%d, ",f, backCell, frontCell);
    dbg(DBG_OFF,"\t%%s => %%s: ", c2XYZ(backCell), c2XYZ(frontCell));
    // On va travailler avec sa backCell
    const int c=backCell>>MD_Shift;
    const int d=backCell &MD_Mask;
    dbg(DBG_OFF,"\t%%d ", c);
    assert(c>=0);
    if (d==(MD_Plus|(MD_DirX+1))) { setFWithTheseNodes(f,c,1,2,5,6); continue; }
    if (d==(MD_Negt|(MD_DirX+1))) { setFWithTheseNodes(f,c,0,3,4,7); continue; }
    if (d==(MD_Plus|(MD_DirY+1))) { setFWithTheseNodes(f,c,2,3,6,7); continue; }
    if (d==(MD_Negt|(MD_DirY+1))) { setFWithTheseNodes(f,c,0,1,4,5); continue; }
    if (d==(MD_Plus|(MD_DirZ+1))) { setFWithTheseNodes(f,c,4,5,6,7); continue; }
    if (d==(MD_Negt|(MD_DirZ+1))) { setFWithTheseNodes(f,c,0,1,2,3); continue; }
    fprintf(stderr,"[nabla_ini_face_node] Error!");
    exit(-1);
    //for(int n=0;n<NABLA_NODE_PER_CELL;n+=1)
    //  dbg(DBG_OFF,"%%d ", cell_node[n*NABLA_NB_CELLS+c]);
  }
  for(int f=0;f<NABLA_NB_FACES;f+=1)
    for(int n=0;n<NABLA_NODE_PER_FACE;++n)
      assert(face_node[n*NABLA_NB_FACES+f]>=0);  
}


// ****************************************************************************
// * Initialisation des coordonnées
// ****************************************************************************
static void nabla_ini_node_coord(void){
  dbgFuncIn();
  dbg(DBG_OFF,"\nasserting NABLA_NB_NODES_Y_AXIS >= 1...");
  assert((NABLA_NB_NODES_Y_AXIS >= 1));

  dbg(DBG_OFF,"\nasserting (NABLA_NB_CELLS %% 1)==0...");
  assert((NABLA_NB_CELLS %% 1)==0);
    
  for(int iNode=0; iNode<NABLA_NB_NODES; iNode+=1){
    const int n=iNode;
    Real x,y,z;
/*#if defined(__MIC__)||defined(__AVX512F__)
    x=set(xOf7(n+7), xOf7(n+6), xOf7(n+5), xOf7(n+4), xOf7(n+3), xOf7(n+2), xOf7(n+1), xOf7(n));
    y=set(yOf7(n+7), yOf7(n+6), yOf7(n+5), yOf7(n+4), yOf7(n+3), yOf7(n+2), yOf7(n+1), yOf7(n));
    z=set(zOf7(n+7), zOf7(n+6), zOf7(n+5), zOf7(n+4), zOf7(n+3), zOf7(n+2), zOf7(n+1), zOf7(n));
#elif __AVX__ || __AVX2__
    x=set(xOf7(n+3), xOf7(n+2), xOf7(n+1), xOf7(n));
    y=set(yOf7(n+3), yOf7(n+2), yOf7(n+1), yOf7(n));
    z=set(zOf7(n+3), zOf7(n+2), zOf7(n+1), zOf7(n));
#elif __SSE2__ && !defined(NO_SSE2)
    x=set(xOf7(n+1), xOf7(n));
    y=set(yOf7(n+1), yOf7(n));
    z=set(zOf7(n+1), zOf7(n));
#else*/
    x=set(xOf7(n));
    y=set(yOf7(n));
    z=set(zOf7(n));
//#endif
    node_coord[iNode]=Real3(x,y,z);
    //dbgReal3(DBG_OFF,node_coord[iNode]);
  }
  //verifCoords();
}
static double xOf7(const int n){
  return ((double)(n%%NABLA_NB_NODES_X_AXIS))*NABLA_NB_NODES_X_TICK;
}
static double yOf7(const int n){
  return ((double)((n/NABLA_NB_NODES_X_AXIS)%%NABLA_NB_NODES_Y_AXIS))*NABLA_NB_NODES_Y_TICK;
}
static double zOf7(const int n){
  return ((double)((n/(NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS))%%NABLA_NB_NODES_Z_AXIS))*NABLA_NB_NODES_Z_TICK;
}
// ****************************************************************************
// * Vérification des coordonnées
// ****************************************************************************
__attribute__((unused)) static void verifCoords(void){
  dbg(DBG_OFF,"\nVérification des coordonnés des noeuds");
  FOR_EACH_NODE(n){
    dbg(DBG_OFF,"\n%%d:",n);
    dbgReal3(DBG_OFF,node_coord[n]);
  }
}


// ****************************************************************************
// * nabla_ini_connectivity
// ****************************************************************************
static void nabla_ini_connectivity(void){
  nabla_ini_node_coord();
  nabla_ini_cell_node();
  nabla_ini_cell_next_prev();
  nabla_ini_node_cell();
  nabla_ini_face_cell();
  nabla_ini_face_node();
  nabla_ini_shift_back_face_cell();
  dbg(DBG_OFF,"\nIni done");
}
