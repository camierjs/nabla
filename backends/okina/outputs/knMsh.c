// ****************************************************************************
// * nabla_ini_node_coords
// ****************************************************************************
static void nabla_ini_node_coords(void){
  dbgFuncIn();
  dbg(DBG_INI,"\nasserting NABLA_NB_NODES_Y_AXIS >= WARP_SIZE...");
  assert((NABLA_NB_NODES_Y_AXIS >= WARP_SIZE));
  //dbg(DBG_INI,"\nasserting (NABLA_NB_NODES_Y_AXIS % WARP_SIZE)==0...");
  //assert((NABLA_NB_NODES_Y_AXIS % WARP_SIZE)==0);
  dbg(DBG_INI,"\nasserting (NABLA_NB_NODES % WARP_SIZE)==0...");
  assert((NABLA_NB_NODES % WARP_SIZE)==0);
  dbg(DBG_INI,"\nasserting (NABLA_NB_CELLS % WARP_SIZE)==0...");
  assert((NABLA_NB_CELLS % WARP_SIZE)==0);
  
  dbg(DBG_INI,"\nOn flush le nombre de mailles attachées à ce noeud");
  FOR_EACH_NODE(n){
    FOR_EACH_NODE_CELL(c){
      node_cell[nc]=-1;
      node_cell_corner[nc]=-1;
    }
  }

  dbg(DBG_INI,"\nOn positionne tous nos noeuds");
  dbg(DBG_INI,"\nNABLA_NB_NODES_X_TICK=%%f",NABLA_NB_NODES_X_TICK);
  dbg(DBG_INI,"\nNABLA_NB_NODES_Y_TICK=%%f",NABLA_NB_NODES_Y_TICK);
  dbg(DBG_INI,"\nNABLA_NB_NODES_Z_TICK=%%f",NABLA_NB_NODES_Z_TICK);

   int iNode=0;
   double dx;
#ifdef __MIC__
#warning __MIC__
  __m512d x,y,z;
  double dy0,dy1,dy2,dy3,dy4,dy5,dy6,dy7;
#elif __AVX__
#warning __AVX__
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
  for(int iX=0;iX<NABLA_NB_NODES_X_AXIS;iX++){
  for(int iZ=0;iZ<NABLA_NB_NODES_Z_AXIS;iZ++){
      for(int iY=0;iY<NABLA_NB_NODES_Y_AXIS;iY+=WARP_SIZE,iNode+=1){
        dx=((double)iX)*NABLA_NB_NODES_X_TICK;
#ifdef __MIC__
        dy0=((double)(iY+0))*NABLA_NB_NODES_Y_TICK;
        dy1=((double)(iY+1))*NABLA_NB_NODES_Y_TICK;
        dy2=((double)(iY+2))*NABLA_NB_NODES_Y_TICK;
        dy3=((double)(iY+3))*NABLA_NB_NODES_Y_TICK;
        dy4=((double)(iY+4))*NABLA_NB_NODES_Y_TICK;
        dy5=((double)(iY+5))*NABLA_NB_NODES_Y_TICK;
        dy6=((double)(iY+6))*NABLA_NB_NODES_Y_TICK;
        dy7=((double)(iY+7))*NABLA_NB_NODES_Y_TICK;
#elif __AVX__
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
#ifdef __MIC__
        x=set(dx,   dx,  dx,  dx,  dx,  dx,  dx,  dx);
        y=set(dy7, dy6, dy5, dy4, dy3, dy2, dy1, dy0);
        z=set(dz,   dz,  dz,  dz,  dz,  dz,  dz,  dz);
#elif __AVX__
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
        //dbg(DBG_INI,"\nSetting nodes-vector #%%d @", iNode);
        //dbgReal3(DBG_INI,node_coord[iNode]);
      }
    }
  }
  
//verifCoords();

  dbg(DBG_INI,"\nOn associe à chaque maille ses noeuds");
  int node_bid,cell_uid,iCell=0;
  /*cilk_*/for(int iX=0;iX<NABLA_NB_CELLS_X_AXIS;iX++){
    /*cilk_*/for(int iZ=0;iZ<NABLA_NB_CELLS_Z_AXIS;iZ++){
      for(int iY=0;iY<NABLA_NB_CELLS_Y_AXIS;iY++,iCell+=1){
        cell_uid=iY+iZ*NABLA_NB_CELLS_Y_AXIS+iX*NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS;
        node_bid=iY+iZ*NABLA_NB_NODES_Y_AXIS+iX*NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS;
        
        dbg(DBG_INI,"\n\tSetting cell #%%d %%dx%%dx%%d, cell_uid=%%d, node_bid=%%d", iCell,iX,iY,iZ,cell_uid,node_bid);
        
        cell_node[0*NABLA_NB_CELLS+iCell] = node_bid;
        cell_node[1*NABLA_NB_CELLS+iCell] = node_bid+1;
        cell_node[2*NABLA_NB_CELLS+iCell] = node_bid  + NABLA_NB_NODES_Y_AXIS+1;
        cell_node[3*NABLA_NB_CELLS+iCell] = node_bid  + NABLA_NB_NODES_Y_AXIS+0;
        cell_node[4*NABLA_NB_CELLS+iCell] = node_bid  + NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS;
        cell_node[5*NABLA_NB_CELLS+iCell] = node_bid  + NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS+1;
        cell_node[6*NABLA_NB_CELLS+iCell] = node_bid  + NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS  +NABLA_NB_NODES_Y_AXIS+1;
        cell_node[7*NABLA_NB_CELLS+iCell] = node_bid  + NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS  +NABLA_NB_NODES_Y_AXIS+0;

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
  
  dbg(DBG_INI,"\nOn associe à chaque maille ses next et prev");



  
  // Pour mimer ARCANE
  // On met des valeurs négatives afin qu le gatherk_and_zero_neg_ones puisse les reconaitre
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirY*NABLA_NB_CELLS+i] = i-1 ;
    cell_next[MD_DirY*NABLA_NB_CELLS+i] = i+1 ;
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    if ((i%NABLA_NB_CELLS_Y_AXIS)==0){
      cell_prev[MD_DirY*NABLA_NB_CELLS+i] = -33333333 ;
      cell_next[MD_DirY*NABLA_NB_CELLS+i+NABLA_NB_CELLS_Y_AXIS-1] = -44444444 ;
    }
  }


  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirZ*NABLA_NB_CELLS+i] = i-NABLA_NB_CELLS_Y_AXIS ;
    cell_next[MD_DirZ*NABLA_NB_CELLS+i] = i+NABLA_NB_CELLS_Y_AXIS ;
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    if ((i%(NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS))<NABLA_NB_CELLS_Y_AXIS){
      cell_prev[MD_DirZ*NABLA_NB_CELLS+i] = -55555555 ;
      cell_next[MD_DirZ*NABLA_NB_CELLS+i+(NABLA_NB_CELLS_Y_AXIS-1)*NABLA_NB_CELLS_Z_AXIS] = -66666666 ;
    }
  }


  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirX*NABLA_NB_CELLS+i] = i-NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS ;
    cell_next[MD_DirX*NABLA_NB_CELLS+i] = i+NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS ;
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    if (i<(NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS)){
      cell_prev[MD_DirX*NABLA_NB_CELLS+i] = -77777777 ;
      cell_next[MD_DirX*NABLA_NB_CELLS+i+(NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Y_AXIS)*(NABLA_NB_CELLS_X_AXIS-1)] = -88888888 ;
    }
  }

    
  dbg(DBG_INI,"\nMaintenant, on re-scan pour remplir la connectivité des noeuds et des coins");
  for(int c=0;c<NABLA_NB_CELLS;c+=1){
    dbg(DBG_INI,"\nFocusing on cells %%d",c);
    //for(int cn=WARP_SIZE*c+WARP_SIZE-1;cn>=WARP_SIZE*c;--cn)
    for(int n=8-1;n>=0;--n){
      const int iNode = cell_node[n*NABLA_NB_CELLS+c];
      //dbg(DBG_INI,"\n\tcell_%%d @%%d:  pushs node %%d",c,n,iNode);
      //int *node=node_cell[iNode];
      //int *corner=node_cell_corner[8*iNode];
      // node_cell a une structure en 1+8 où le 1 donne le nombre de mailles à ce noeud
      // les 8 autres emplacements donnent l'offset jusqu'aux mailles
      // node_corner a une structure en 8*NABLA_NB_NODES
      //node[0]+=1;
      //node[node[0]]=cn;
      node_cell[8*iNode+n]=c;
      //corner[0]+=1;
      //corner[corner[0]]=n;
      node_cell_corner[8*iNode+n]=n;
   }
  }

  /*dbg(DBG_INI,"\nVérification des connectivité des noeuds");
  FOR_EACH_NODE(n){
    dbg(DBG_INI,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL(c){
      dbg(DBG_INI,"\n\tnode_%%d knows cell %%d",n,node_cell[nc]);
    }
  }
  dbg(DBG_INI,"\nVérification des coins des noeuds");
  FOR_EACH_NODE(n){
    dbg(DBG_INI,"\nFocusing on node %%d",n);
    FOR_EACH_NODE_CELL(c){
      if (node_cell_corner[nc]!=-1)
        dbg(DBG_INI,"\n\tnode_%%d is corner #%%d of cell %%d",n,node_cell_corner[nc],node_cell[nc]);
    }
    }*/
  dbg(DBG_INI,"\nIni done");
  dbgFuncOut();
}


__attribute__((unused)) static void verifCoords(void){
  dbg(DBG_INI,"\nVérification des coordonnés des noeuds");
  FOR_EACH_NODE_WARP(n){
    dbg(DBG_INI,"\nFocusing on nodes-vector %%d",n);
#ifdef _OKINA_SOA_
    dbgReal(DBG_INI,node_coordx[n]);
    dbgReal(DBG_INI,node_coordy[n]);
    dbgReal(DBG_INI,node_coordz[n]);
#else
    dbgReal3(DBG_INI,node_coord[n]);
#endif
  }
}
