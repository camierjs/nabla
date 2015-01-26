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
#include "nabla.h"


void okinaDefineEnumerates(nablaMain *nabla){
  const char *parallel_prefix_for_loop=nabla->parallel->loop(nabla);
  fprintf(nabla->entity->hdr,"\n\n\
/*********************************************************\n\
 * Forward enumerates\n\
 *********************************************************/\n\
#define FOR_EACH_CELL(c) %sfor(int c=0;c<NABLA_NB_CELLS;c+=1)\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<8;n+=1)\n\
\n\
#define FOR_EACH_CELL_WARP(c) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
#define FOR_EACH_CELL_WARP_SHARED(c,local) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
\n\
#define FOR_EACH_CELL_WARP_NODE(n)\\\n\
  %sfor(int cn=WARP_SIZE*c+WARP_SIZE-1;cn>=WARP_SIZE*c;--cn)\\\n\
    for(int n=8-1;n>=0;--n)\n\
\n\
#define FOR_EACH_NODE(n) /*%s*/for(int n=0;n<NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE_CELL(c) for(int c=0,nc=8*n;c<8;c+=1,nc+=1)\n\
\n\
#define FOR_EACH_NODE_WARP(n) %sfor(int n=0;n<NABLA_NB_NODES_WARP;n+=1)\n\
\n\
#define FOR_EACH_NODE_WARP_CELL(c)\\\n\
    for(int c=0;c<8;c+=1)\n",
          parallel_prefix_for_loop, // FOR_EACH_CELL
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_SHARED
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_NODE
          parallel_prefix_for_loop, // FOR_EACH_NODE
          parallel_prefix_for_loop  // FOR_EACH_NODE_WARP
          );
}
