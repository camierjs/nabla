// ****************************************************************************
// * nabla_ini_node_coords
// ****************************************************************************
static void verifCorners(void);
static void verifNextPrev(void);
static void verifConnectivity(void);

static int comparNodeCell(const void *a, const void *b){
  return (*(int*)a)>(*(int*)b);
}

static int comparNodeCellAndCorner(const void *pa, const void *pb){
  int *a=(int*)pa;
  int *b=(int*)pb;
  return a[0]>b[0];
}

static void nabla_ini_node_coords(void){
  dbgFuncIn();
  dbg(DBG_INI,"\nasserting NABLA_NB_NODES_Y_AXIS >= WARP_SIZE...");
  assert((NABLA_NB_NODES_Y_AXIS >= WARP_SIZE));
  //dbg(DBG_INI,"\nasserting (NABLA_NB_NODES_Y_AXIS %% WARP_SIZE)==0...");
  //assert((NABLA_NB_NODES_Y_AXIS %% WARP_SIZE)==0);
  dbg(DBG_INI,"\nasserting (NABLA_NB_NODES %% WARP_SIZE)==0...");
  assert((NABLA_NB_NODES %% WARP_SIZE)==0);
  dbg(DBG_INI,"\nasserting (NABLA_NB_CELLS %% WARP_SIZE)==0...");
  assert((NABLA_NB_CELLS %% WARP_SIZE)==0);
  
  dbg(DBG_INI,"\nOn positionne tous nos noeuds");
  dbg(DBG_INI,"\nNABLA_NB_NODES_X_TICK=%%f",NABLA_NB_NODES_X_TICK);
  dbg(DBG_INI,"\nNABLA_NB_NODES_Y_TICK=%%f",NABLA_NB_NODES_Y_TICK);
  dbg(DBG_INI,"\nNABLA_NB_NODES_Z_TICK=%%f",NABLA_NB_NODES_Z_TICK);

  #warning HW tied to undef __SSE2__
#undef __SSE2__
  
   int iNode=0;
   double dx;
#if defined(__MIC__)||defined(__AVX512F__)
#warning __MIC__||__AVX512F__
  __m512d x,y,z;
  double dy0,dy1,dy2,dy3,dy4,dy5,dy6,dy7;
#elif defined(__AVX__) || defined(__AVX2__)
   #ifdef __AVX2__
      #warning __AVX2__
   #elif __AVX__
      #warning __AVX__
   #endif
   __m256d x,y,z;
   double dy0,dy1,dy2,dy3;
#elif __SSE2__
#warning __SSE2__
   __m128d x,y,z;
   double dy0,dy1;
#else
#warning __NO_SSE_AVX_MIC__
   double x,y,z;
   double dy0;
#endif
  double dz;
  for(int iZ=0;iZ<NABLA_NB_NODES_Z_AXIS;iZ++){
    for(int iY=0;iY<NABLA_NB_NODES_Y_AXIS;iY++){
      for(int iX=0;iX<NABLA_NB_NODES_X_AXIS;iX+=WARP_SIZE,iNode+=1){
        dx=((double)iX)*NABLA_NB_NODES_X_TICK;
#if defined(__MIC__)||defined(__AVX512F__)
        dy0=((double)(iY+0))*NABLA_NB_NODES_Y_TICK;
        dy1=((double)(iY+1))*NABLA_NB_NODES_Y_TICK;
        dy2=((double)(iY+2))*NABLA_NB_NODES_Y_TICK;
        dy3=((double)(iY+3))*NABLA_NB_NODES_Y_TICK;
        dy4=((double)(iY+4))*NABLA_NB_NODES_Y_TICK;
        dy5=((double)(iY+5))*NABLA_NB_NODES_Y_TICK;
        dy6=((double)(iY+6))*NABLA_NB_NODES_Y_TICK;
        dy7=((double)(iY+7))*NABLA_NB_NODES_Y_TICK;
#elif __AVX__||__AVX2__
        dy0=((double)(iY+0))*NABLA_NB_NODES_Y_TICK;
        dy1=((double)(iY+1))*NABLA_NB_NODES_Y_TICK;
        dy2=((double)(iY+2))*NABLA_NB_NODES_Y_TICK;
        dy3=((double)(iY+3))*NABLA_NB_NODES_Y_TICK;
#elif __SSE2__
        dy0=((double)(iY+0))*NABLA_NB_NODES_Y_TICK;
        dy1=((double)(iY+1))*NABLA_NB_NODES_Y_TICK;
#else
        dy0=((double)(iY+0))*NABLA_NB_NODES_Y_TICK;
#endif
        dz=((double)iZ)*NABLA_NB_NODES_Z_TICK;
        // On calcule les coordonnées du vecteur de points
#if defined(__MIC__)||defined(__AVX512F__)
        x=set(dx,   dx,  dx,  dx,  dx,  dx,  dx,  dx);
        y=set(dy7, dy6, dy5, dy4, dy3, dy2, dy1, dy0);
        z=set(dz,   dz,  dz,  dz,  dz,  dz,  dz,  dz);
#elif __AVX__ || __AVX2__
        x=set(dx,   dx,  dx,  dx);
        y=set(dy3, dy2, dy1, dy0);
        z=set(dz,   dz,  dz,  dz);
#elif __SSE2__
        x=set(dx,   dx);
        y=set(dy1, dy0);
        z=set(dz,   dz);
#else
        x=set(dx);
        y=set(dy0);
        z=set(dz);
#endif
        // Là où l'on poke le retour de okinaSourceMeshAoS_vs_SoA
        %s
        //node_coord[iNode]=Real3(x,y,z);
        //dbg(DBG_INI,"\nSetting nodes-vector #%%d @", iNode);
        //dbgReal3(DBG_INI,node_coord[iNode]);
      }
    }
  }
  //verifCoords();

  dbg(DBG_INI,"\nOn associe à chaque maille ses noeuds");
  int node_bid,cell_uid,iCell=0;
  for(int iZ=0;iZ<NABLA_NB_CELLS_Z_AXIS;iZ++){
    for(int iY=0;iY<NABLA_NB_CELLS_Y_AXIS;iY++){
      for(int iX=0;iX<NABLA_NB_CELLS_X_AXIS;iX++,iCell+=1){
        cell_uid=iX + iY*NABLA_NB_CELLS_X_AXIS + iZ*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS;
        node_bid=iX + iY*NABLA_NB_NODES_X_AXIS + iZ*NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;
        dbg(DBG_INI,"\n\tSetting cell #%%d %%dx%%dx%%d, cell_uid=%%d, node_bid=%%d", iCell,iX,iY,iZ,cell_uid,node_bid);
        cell_node[0*NABLA_NB_CELLS+iCell] = node_bid;
        cell_node[1*NABLA_NB_CELLS+iCell] = node_bid + 1;
        cell_node[2*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS + 1;
        cell_node[3*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS + 0;
        cell_node[4*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;
        cell_node[5*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS +1 ;
        cell_node[6*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS + NABLA_NB_NODES_X_AXIS+1;
        cell_node[7*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS + NABLA_NB_NODES_X_AXIS;
        dbg(DBG_INI,"\n\tCell_%%d's nodes are %%d,%%d,%%d,%%d,%%d,%%d,%%d,%%d", iCell,
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
  dbg(DBG_INI,"\nMaintenant, on re-scan pour remplir la connectivité des noeuds et des coins");
  dbg(DBG_INI,"\nOn flush le nombre de mailles attachées à ce noeud");
  /*FOR_EACH_NODE(n){
    FOR_EACH_NODE_CELL(c){
      node_cell[nc]=-1;
      node_cell_corner[nc]=-1;
    }
    }*/
  for(int n=0;n<NABLA_NB_NODES;n+=1){
    for(int c=0;c<8;++c){
      node_cell[8*n+c]=-1;
      node_cell_corner[8*n+c]=-1;
      node_cell_and_corner[2*(8*n+c)+0]=-1;//cell
      node_cell_and_corner[2*(8*n+c)+1]=-1;//corner
    }
  }
  for(int c=0;c<NABLA_NB_CELLS;c+=1){
    dbg(DBG_INI,"\nFocusing on cells %%d",c);
    for(int n=0;n<8;n++){
      const int iNode = cell_node[n*NABLA_NB_CELLS+c];
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
  //                          int (*compar)(const void *, const void *));
  for(int n=0;n<NABLA_NB_NODES;n+=1){
    qsort(&node_cell[8*n],8,sizeof(int),comparNodeCell);
    qsort(&node_cell_and_corner[2*8*n],8,2*sizeof(int),comparNodeCellAndCorner);
  }
  // And we come back to set our node_cell_corner
  for(int n=0;n<NABLA_NB_NODES;n+=1)
    for(int c=0;c<8;++c)
      node_cell_corner[8*n+c]=node_cell_and_corner[2*(8*n+c)+1];

  verifConnectivity();
  verifCorners();

  
  dbg(DBG_INI,"\nOn associe à chaque maille ses next et prev");
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
      cell_next[MD_DirY*NABLA_NB_CELLS+i+(NABLA_NB_CELLS_X_AXIS-1)*NABLA_NB_CELLS_Y_AXIS] = -66666666 ;
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
      cell_next[MD_DirZ*NABLA_NB_CELLS+i+(NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS)*(NABLA_NB_CELLS_Z_AXIS-1)] = -88888888 ;
    }
  }
  verifNextPrev(); 
  dbg(DBG_INI,"\nIni done");
  dbgFuncOut();
}


__attribute__((unused)) static void verifCoords(void){
  dbg(DBG_INI,"\nVérification des coordonnés des noeuds");
  FOR_EACH_NODE_WARP(n){
    //dbg(DBG_INI,"\nFocusing on nodes-vector %%d",n);
#ifdef _OKINA_SOA_
    dbgReal(DBG_INI,node_coordx[n]);
    dbgReal(DBG_INI,node_coordy[n]);
    dbgReal(DBG_INI,node_coordz[n]);
#else
    dbgReal3(DBG_INI,node_coord[n]);
#endif
  }
}


__attribute__((unused)) static void verifConnectivity(void){
  dbg(DBG_INI,"\nVérification des connectivité des noeuds");
  FOR_EACH_NODE(n){
    dbg(DBG_INI,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL(c){
      dbg(DBG_INI,"\n\tnode_%%d knows cell %%d",n,node_cell[nc]);
      dbg(DBG_INI,", and node_%%d knows cell %%d",n,node_cell_and_corner[2*nc+0]);
    }
  }
}

__attribute__((unused)) static void verifCorners(void){
  dbg(DBG_INI,"\nVérification des coins des noeuds");
  FOR_EACH_NODE(n){
    dbg(DBG_INI,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL(c){
      //if (node_cell_corner[nc]!=-1)
      dbg(DBG_INI,"\n\tnode_%%d is corner #%%d of cell %%d",n,node_cell_corner[nc],node_cell[nc]);
      dbg(DBG_INI,", and node_%%d is corner #%%d of cell %%d",n,node_cell_and_corner[2*nc+1],node_cell_and_corner[2*nc+0]);
    }
  }
}

__attribute__((unused)) static void verifNextPrev(void){
 for (int i=0; i<NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI,"\nNext/Prev(X) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirX*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirX*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI,"\nNext/Prev(Y) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirY*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirY*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI,"\nNext/Prev(Z) for cells %%d <- #%%d -> %%d: ",
        cell_prev[MD_DirZ*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirZ*NABLA_NB_CELLS+i]);
  }
}
