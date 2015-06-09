#include "gram.h"



// ********************************************************
// * loop job
// ********************************************************
static inline void loop(/* direct return from okinaHookAddExtraParameters*/){
	dbgFuncIn();
	FOR_EACH_NODE_WARP(n){
		{
		/* DiffractingREADY *//*isLeft*/unp1 /*'='->!isLeft*/=opSub ( opSub ( /*NodeJob*//*tt2a*/node_u/*nvar no diffraction possible here*//*NodeVar !20*/[n], opMul ( cp , ( opSub ( /*NodeJob*//*tt2a*/node_u/*nvar no diffraction possible here*//*NodeVar !20*/[n], /*no_item_system*//*NodeJob*//*tt2a*/node_u/*nvar no diffraction possible here*//*NodeVar 20*/[- 1 ]) ) ) ) , opMul ( cm , ( opSub ( /*no_item_system*//*NodeJob*//*tt2a*/node_u/*nvar no diffraction possible here*//*NodeVar 20*/[+ 1 ], /*NodeJob*//*tt2a*/node_u/*nvar no diffraction possible here*//*NodeVar !20*/[n]) ) ) ) ;
		}}
}
// ******************************************************************************
// * Kernel d'initialisation des variables
// ******************************************************************************
void nabla_ini_variables(void){
	FOR_EACH_NODE_WARP(n){
		node_u[n]=zero();
	}
	FOR_EACH_CELL_WARP(c){
		cell_alpha[c]=zero();
	}
}


// ******************************************************************************
// * Main d'Okina
// ******************************************************************************
int main(int argc, char *argv[]){
	float cputime=0.0;
	struct timeval st, et;
	//int iteration=1;
#ifdef __AVX__
	//avxTest();
#endif
#if defined(__MIC__)||defined(__AVX512F__)
	//micTestReal();
	//micTestReal3();
#endif
	printf("%d noeuds, %d mailles",NABLA_NB_NODES,NABLA_NB_CELLS);
	nabla_ini_variables();
	nabla_ini_node_coords();
	// Initialisation de la pr�cision du cout
	std::cout.precision(21);
	//std::cout.setf(std::ios::floatfield);
	std::cout.setf(std::ios::scientific, std::ios::floatfield);
	// Initialisation du temps et du deltaT
	global_time=0.0;
	global_iteration=1;
	global_deltat[0] = set1(option_dtt_initial);// @ 0;
	//printf("\n\33[7;32m[main] time=%e, Global Iteration is #%d\33[m",global_time,global_iteration);
	// okinaGenerateSingleVariable
	// okinaGenerateSingleVariableMalloc
	// okinaGenerateSingleVariable
	// okinaGenerateSingleVariableMalloc
	// okinaGenerateSingleVariable
	// okinaGenerateSingleVariableMalloc
	// okinaGenerateSingleVariable
	// okinaGenerateSingleVariableMalloc	// [nccOkinaMainMeshPrefix] Allocation des connectivit�s
	//OKINA_MAIN_PREINIT
	//printf("\ndbgsVariable iteration"); dbgCellVariableDim0_iteration();
	//printf("\ndbgsVariable alpha"); dbgCellVariableDim0_alpha();
	//printf("\ndbgsVariable u"); dbgNodeVariableDim0_u();
	//printf("\ndbgsVariable alpha_global"); dbgCellVariableDim0_alpha_global();
	gettimeofday(&st, NULL);
	while (global_time<option_stoptime){// && global_iteration!=option_max_iterations){
		
		/*@1.000000*/loop(
		/*okinaAddExtraArguments*/
		/*okinaDumpNablaArgumentList*//*NULL_called_variables*/);
		/*okinaDumpNablaDebugFunctionFromOutArguments*/
	//OKINA_MAIN_POSTINIT
	// okinaGenerateSingleVariableFree
	// okinaGenerateSingleVariableFree
	// okinaGenerateSingleVariableFree
	// okinaGenerateSingleVariableFree
//OKINA_MAIN_POSTFIX
	global_time+=*(double*)&global_deltat[0];
	global_iteration+=1;
	//printf("\ntime=%e, dt=%e\n", global_time, *(double*)&global_deltat[0]);
	}	gettimeofday(&et, NULL);
	cputime = ((et.tv_sec-st.tv_sec)*1000.+ (et.tv_usec - st.tv_usec)/1000.0);
	printf("\n\t\33[7m[#%04d] Elapsed time = %12.6e(s)\33[m\n", global_iteration-1, cputime/1000.0);

}
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
// * nabla_ini_node_coords
// ****************************************************************************
static void verifCoords(void);
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

static double xOf7(const int n){
  return ((double)(n%NABLA_NB_NODES_X_AXIS))*NABLA_NB_NODES_X_TICK;
}
static double yOf7(const int n){
  return ((double)((n/NABLA_NB_NODES_X_AXIS)%NABLA_NB_NODES_Y_AXIS))*NABLA_NB_NODES_Y_TICK;
}
static double zOf7(const int n){
  return ((double)((n/(NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS))%NABLA_NB_NODES_Z_AXIS))*NABLA_NB_NODES_Z_TICK;
}

static void nabla_ini_node_coords(void){
  dbgFuncIn();
  dbg(DBG_INI,"\nasserting NABLA_NB_NODES_Y_AXIS >= WARP_SIZE...");
  assert((NABLA_NB_NODES_Y_AXIS >= WARP_SIZE));

  dbg(DBG_INI,"\nasserting (NABLA_NB_CELLS % WARP_SIZE)==0...");
  assert((NABLA_NB_CELLS % WARP_SIZE)==0);
    
  for(int iNode=0; iNode<NABLA_NB_NODES_WARP; iNode+=1){
    const int n=WARP_SIZE*iNode;
    Real x,y,z;
#if defined(__MIC__)||defined(__AVX512F__)
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
#else
    x=set(xOf7(n));
    y=set(yOf7(n));
    z=set(zOf7(n));
#endif
    // Là où l'on poke le retour de okinaSourceMeshAoS_vs_SoA
    node_coord[iNode]=Real3(x,y,z);
    //node_coord[n]=Real3(x,y,z);
    //dbg(DBG_INI,"\nSetting nodes-vector #%d @", n);
    dbgReal3(DBG_INI,node_coord[iNode]);
  }
  verifCoords();

  dbg(DBG_INI,"\nOn associe à chaque maille ses noeuds");
  int node_bid,cell_uid,iCell=0;
  for(int iZ=0;iZ<NABLA_NB_CELLS_Z_AXIS;iZ++){
    for(int iY=0;iY<NABLA_NB_CELLS_Y_AXIS;iY++){
      for(int iX=0;iX<NABLA_NB_CELLS_X_AXIS;iX++,iCell+=1){
        cell_uid=iX + iY*NABLA_NB_CELLS_X_AXIS + iZ*NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS;
        node_bid=iX + iY*NABLA_NB_NODES_X_AXIS + iZ*NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;
        dbg(DBG_INI,"\n\tSetting cell #%d %dx%dx%d, cell_uid=%d, node_bid=%d",
            iCell,iX,iY,iZ,cell_uid,node_bid);
        cell_node[0*NABLA_NB_CELLS+iCell] = node_bid;
        cell_node[1*NABLA_NB_CELLS+iCell] = node_bid + 1;
        cell_node[2*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS + 1;
        cell_node[3*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS + 0;
        cell_node[4*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS;
        cell_node[5*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS +1 ;
        cell_node[6*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS + NABLA_NB_NODES_X_AXIS+1;
        cell_node[7*NABLA_NB_CELLS+iCell] = node_bid + NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS + NABLA_NB_NODES_X_AXIS;
        dbg(DBG_INI,"\n\tCell_%d's nodes are %d,%d,%d,%d,%d,%d,%d,%d", iCell,
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

  for(int n=0;n<NABLA_NB_NODES;n+=1){
    for(int c=0;c<8;++c){
      node_cell[8*n+c]=-1;
      node_cell_corner[8*n+c]=-1;
      node_cell_and_corner[2*(8*n+c)+0]=-1;//cell
      node_cell_and_corner[2*(8*n+c)+1]=-1;//corner
    }
  }
  
  for(int c=0;c<NABLA_NB_CELLS;c+=1){
    dbg(DBG_INI,"\nFocusing on cells %d",c);
    for(int n=0;n<8;n++){
      const int iNode = cell_node[n*NABLA_NB_CELLS+c];
      dbg(DBG_INI,"\n\tcell_%d @%d: pushs node %d",c,n,iNode);
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

  //verifConnectivity();
  verifCorners();
  
  dbg(DBG_INI,"\nOn associe à chaque maille ses next et prev");
  // On met des valeurs négatives afin que le gatherk_and_zero_neg_ones puisse les reconaitre
  // Dans la direction X
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    cell_prev[MD_DirX*NABLA_NB_CELLS+i] = i-1 ;
    cell_next[MD_DirX*NABLA_NB_CELLS+i] = i+1 ;
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    if ((i%NABLA_NB_CELLS_X_AXIS)==0){
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
    if ((i%(NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS))<NABLA_NB_CELLS_Y_AXIS){
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
    // dbg(DBG_INI,"\nFocusing on nodes-vector %d",n);
    // _OKINA_SOA_ is defined or not depending on nabla's colors
#ifdef _OKINA_SOA_
    dbgReal(DBG_INI,node_coordx[n]);
    dbgReal(DBG_INI,node_coordy[n]);
    dbgReal(DBG_INI,node_coordz[n]);
#else
    dbg(DBG_INI,"\n%d:",n);
    dbgReal3(DBG_INI,node_coord[n]);
#endif
  }
}


__attribute__((unused)) static void verifConnectivity(void){
  dbg(DBG_INI,"\nVérification des connectivité des noeuds");
  FOR_EACH_NODE(n){
    dbg(DBG_INI,"\nFocusing on node %d",n);
    FOR_EACH_NODE_CELL(c){
      dbg(DBG_INI,"\n\tnode_%d knows cell %d",n,node_cell[nc]);
      dbg(DBG_INI,", and node_%d knows cell %d",n,node_cell_and_corner[2*nc+0]);
    }
  }
}

__attribute__((unused)) static void verifCorners(void){
  dbg(DBG_INI,"\nVérification des coins des noeuds");
  FOR_EACH_NODE(n){
    dbg(DBG_INI,"\nFocusing on node %d",n);
    FOR_EACH_NODE_CELL(c){
      if (node_cell_corner[nc]==-1) continue;
      dbg(DBG_INI,"\n\tnode_%d is corner #%d of cell %d",n,node_cell_corner[nc],node_cell[nc]);
      //dbg(DBG_INI,", and node_%d is corner #%d of cell %d",n,node_cell_and_corner[2*nc+1],node_cell_and_corner[2*nc+0]);
    }
  }
}

__attribute__((unused)) static void verifNextPrev(void){
 for (int i=0; i<NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI,"\nNext/Prev(X) for cells %d <- #%d -> %d: ",
        cell_prev[MD_DirX*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirX*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI,"\nNext/Prev(Y) for cells %d <- #%d -> %d: ",
        cell_prev[MD_DirY*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirY*NABLA_NB_CELLS+i]);
  }
  for (int i=0; i<NABLA_NB_CELLS; ++i) {
    dbg(DBG_INI,"\nNext/Prev(Z) for cells %d <- #%d -> %d: ",
        cell_prev[MD_DirZ*NABLA_NB_CELLS+i],
        i,
        cell_next[MD_DirZ*NABLA_NB_CELLS+i]);
  }
}
