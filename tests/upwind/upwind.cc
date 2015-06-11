#include "upwind.h"


// ********************************************************
// * iniGlobals fct
// ********************************************************
static inline void iniGlobals(/*numParams=0*/){
	dbgFuncIn();
	
	global_dtx[0]=opDiv((opSub(xmax,xmin)),X_EDGE_ELEMS);
	global_deltat[0]=opDiv(opMul(CFL,global_dtx[0]),option_a);
	/*printf*/printf("\n\t\t\33[7m[iniGlobals] dtx=%f\33[m",global_dtx[0]);
	/*printf*/printf("\n\t\t\33[7m[iniGlobals] dtt=%f\33[m",global_deltat[0]);
	/*assert*/assert(global_deltat[0]>=0.0);
	}



// ********************************************************
// * inidtx job
// ********************************************************
static inline void inidtx(){
	dbgFuncIn();
	FOR_EACH_NODE(n){
		{
		if ( /*real3*//*if+real3 still in real3 vs double3*/node_coord[n]!= xmin) /*real*/((double*)node_dtxm)[n]=opSub ( /*real3*//*if+real3 still in real3 vs double3*/node_coord[n], /*real3*//*if+real3 still in real3 vs double3*/node_coord[- 1 ]) ;
		if ( /*real3*//*if+real3 still in real3 vs double3*/node_coord[n]!= xmax) /*real*/((double*)node_dtxp)[n]=opSub ( /*real3*//*if+real3 still in real3 vs double3*/node_coord[+ 1 ], /*real3*//*if+real3 still in real3 vs double3*/node_coord[n]) ;
		}}
}


// ********************************************************
// * iniU job
// ********************************************************
static inline void iniU(){
	dbgFuncIn();
	FOR_EACH_NODE_WARP(n){
		{
		node_u[n]=0.0 ;
		node_u[n]=opTernary ( ( test== 1 ) , u0_Test1_for_linear_advection_smooth_data ( node_coord[n]) , node_u[n]) ;
		node_u[n]=opTernary ( ( test== 2 ) , u0_Test2_for_linear_advection_discontinuous_data ( node_coord[n]) , node_u[n]) ;
		}}
}

// ********************************************************
// * dbgLoop fct
// ********************************************************
static inline void dbgLoop(/*numParams=0*/){
	dbgFuncIn();
	
	/*printf*/printf("\n\t\t\33[7m[Loop] #%d, time=%f\33[m",GlobalIteration,global_time);
	}



// ********************************************************
// * loop job
// ********************************************************
static inline void loop(){
	dbgFuncIn();
	FOR_EACH_NODE_WARP(n){
		{
		__attribute__ ((aligned(WARP_ALIGN))) const real ap =fmax ( option_a, 0.0 ) ;
		__attribute__ ((aligned(WARP_ALIGN))) const real am =fmin ( option_a, 0.0 ) ;
		__attribute__ ((aligned(WARP_ALIGN))) const real dttSx =opDiv ( global_deltat[0]/*n g*/, global_dtx[0]/*n g*/) ;
		__attribute__ ((aligned(WARP_ALIGN))) const real cp =opMul ( ap , dttSx ) ;
		__attribute__ ((aligned(WARP_ALIGN))) const real cm =opMul ( am , dttSx ) ;
		node_unp1[n]=opSub ( opSub ( node_u[n], opMul ( cp , ( opSub ( node_u[n], node_u[opSub ( n , 1 ) ]) ) ) ) , opMul ( cm , ( opSub ( node_u[opAdd ( n , 1 ) ], node_u[n]) ) ) ) ;
		}}
}


// ********************************************************
// * copyResults job
// ********************************************************
static inline void copyResults(){
	dbgFuncIn();
	FOR_EACH_NODE_WARP(n){
		{
		node_u[n]=node_unp1[n];
		}}
}


// ********************************************************
// * dumpSolution job
// ********************************************************
static inline void dumpSolution(File results ){
	dbgFuncIn();
	FOR_EACH_NODE_WARP(n){
		{
		results <<node_coord[n]<<"\t" <<node_u[n]<<"\t" <<node_unp1[n]<<"\n" ;
		}}
}

// ********************************************************
// * tstForQuit fct
// ********************************************************
static inline void tstForQuit(/*numParams=0*/){
	dbgFuncIn();
	
	/*printf*/printf("\n\t[testForQuit] GlobalIteration =%d",GlobalIteration);
	if ( GlobalIteration<time_steps) return ;
	{
file ( results , plot ) ;
	/*printf*/printf("\n\t[testForQuit] GlobalIteration>time_steps, dumping\n");
	/*dumpSolution*/dumpSolution(results);
	}
exit(0.0);
	}


// ********************************************************
// * u0_Test1_for_linear_advection_smooth_data fct
// ********************************************************
static inline Real u0_Test1_for_linear_advection_smooth_data(Real x /*numParams=1*/){
	dbgFuncIn();
	
	return opMul(al,/*exp*/exp(opMul(-bt,pow(x,2.0))));
	}


// ********************************************************
// * u0_Test2_for_linear_advection_discontinuous_data fct
// ********************************************************
static inline Real u0_Test2_for_linear_advection_discontinuous_data(Real x /*numParams=1*/){
	dbgFuncIn();
	
	if ( x<0.3) return 0.0;
	if ( x<0.7) return 1.0;
	return 0.0;
	}

// ******************************************************************************
// * Kernel d'initialisation des variables
// ******************************************************************************
void nabla_ini_variables(void){
	FOR_EACH_NODE_WARP(n){
		node_u[n]=zero();
		node_unp1[n]=zero();
		node_dtxp[n]=zero();
		node_dtxm[n]=zero();
	}
	FOR_EACH_CELL_WARP(c){
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
	//printf("\n\33[7;32m[main] time=%e, Global Iteration is #%d\33[m",global_time,global_iteration);	// [nOkinaMainMeshPrefix] Allocation des connectivit�s
	//OKINA_MAIN_PREINIT
	//printf("\ndbgsVariable iteration"); dbgCellVariableDim0_iteration();
	//printf("\ndbgsVariable u"); dbgNodeVariableDim0_u();
	//printf("\ndbgsVariable unp1"); dbgNodeVariableDim0_unp1();
	//printf("\ndbgsVariable dtxp"); dbgNodeVariableDim0_dtxp();
	//printf("\ndbgsVariable dtxm"); dbgNodeVariableDim0_dtxm();
	//printf("\ndbgsVariable dtx"); dbgCellVariableDim0_dtx();
	/*@-5.000000*/iniGlobals(/*NULL_nblParamsNode*//*NULL_called_variables*/);
	/*@-5.000000*/inidtx(/*NULL_called_variables*/);
	
	/*@-4.000000*/iniU(/*NULL_called_variables*/);
	gettimeofday(&st, NULL);
	while (global_time<option_stoptime){// && global_iteration!=option_max_iterations){
		
		/*@1.000000*/dbgLoop(/*NULL_nblParamsNode*//*NULL_called_variables*/);
		/*@1.000000*/loop(/*NULL_called_variables*/);
		
		/*@2.000000*/copyResults(/*NULL_called_variables*/);
		
		/*@4.000000*/tstForQuit(/*NULL_nblParamsNode*//*NULL_called_variables*/);
	//OKINA_MAIN_POSTINIT
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
// * 1D
// ****************************************************************************

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
//static double yOf7(const int n){ return 0.0; }
//static double zOf7(const int n){ return 0.0; }

static void nabla_ini_node_coords(void){
  dbgFuncIn();

  dbg(DBG_INI,"\nasserting (NABLA_NB_CELLS % WARP_SIZE)==0...");
  assert((NABLA_NB_CELLS % WARP_SIZE)==0);
    
  for(int iNode=0; iNode<NABLA_NB_NODES_WARP; iNode+=1){
    const int n=WARP_SIZE*iNode;
    Real x;
#if defined(__MIC__)||defined(__AVX512F__)
    x=set(xOf7(n+7), xOf7(n+6), xOf7(n+5), xOf7(n+4), xOf7(n+3), xOf7(n+2), xOf7(n+1), xOf7(n));
#elif __AVX__ || __AVX2__
    x=set(xOf7(n+3), xOf7(n+2), xOf7(n+1), xOf7(n));
#elif __SSE2__ && !defined(NO_SSE2)
    x=set(xOf7(n+1), xOf7(n));
#else
    x=set(xOf7(n));
#endif
    // Là où l'on poke le retour de okinaSourceMeshAoS_vs_SoA
    node_coord[iNode]=Real(x);
    dbg(DBG_INI,"\nSetting nodes-vector #%d @", n);
    dbgReal(DBG_INI,node_coord[iNode]);
  }
  verifCoords();

  dbg(DBG_INI,"\nOn associe à chaque maille ses noeuds");
  int node_bid,cell_uid,iCell=0;
  for(int iX=0;iX<NABLA_NB_CELLS_X_AXIS;iX++,iCell+=1){
    cell_uid=iX;
    node_bid=iX;
    dbg(DBG_INI,"\n\tSetting cell #%d, cell_uid=%d, node_bid=%d",
        iCell,cell_uid,node_bid);
    cell_node[0*NABLA_NB_CELLS+iCell] = node_bid;
    cell_node[1*NABLA_NB_CELLS+iCell] = node_bid + 1;
    dbg(DBG_INI,"\n\tCell_%d's nodes are %d,%d", iCell,
        cell_node[0*NABLA_NB_CELLS+iCell],
        cell_node[1*NABLA_NB_CELLS+iCell]);
  }
  dbg(DBG_INI,"\nMaintenant, on re-scan pour remplir la connectivité des noeuds et des coins");
  dbg(DBG_INI,"\nOn flush le nombre de mailles attachées à ce noeud");

  for(int n=0;n<NABLA_NB_NODES;n+=1){
    for(int c=0;c<2;++c){
      node_cell[2*n+c]=-1;
      node_cell_corner[2*n+c]=-1;
      node_cell_and_corner[2*(2*n+c)+0]=-1;//cell
      node_cell_and_corner[2*(2*n+c)+1]=-1;//corner
    }
  }
  
  for(int c=0;c<NABLA_NB_CELLS;c+=1){
    dbg(DBG_INI,"\nFocusing on cells %d",c);
    for(int n=0;n<2;n++){
      const int iNode = cell_node[n*NABLA_NB_CELLS+c];
      dbg(DBG_INI,"\n\tcell_%d @%d: pushs node %d",c,n,iNode);
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
  for(int n=0;n<NABLA_NB_NODES;n+=1){
    qsort(&node_cell[2*n],2,sizeof(int),comparNodeCell);
    qsort(&node_cell_and_corner[2*2*n],2,2*sizeof(int),comparNodeCellAndCorner);
  }
  // And we come back to set our node_cell_corner
  for(int n=0;n<NABLA_NB_NODES;n+=1)
    for(int c=0;c<2;++c)
      node_cell_corner[2*n+c]=node_cell_and_corner[2*(2*n+c)+1];

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
    dbgReal(DBG_INI,node_coord[n]);
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
}
