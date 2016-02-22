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
#include "nabla.h"
//#include "backends/kokkos/.h"
#include "backends/lib/dump/dump.h"


// ****************************************************************************
// * hookHeaderOpen
// ****************************************************************************
void hookHeaderOpen(nablaMain *nabla){
  char hdrFileName[NABLA_MAX_FILE_NAME];
  // Ouverture du fichier header
  sprintf(hdrFileName, "%s.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
}


// ****************************************************************************
// * hookHeaderIncludes
// ****************************************************************************
void hookHeaderIncludes(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\n\
// *****************************************************************************\n\
// * Backend includes\n\
// *****************************************************************************\n\
// Includes from nabla->simd->includes\n\%s\
#include <sys/time.h>\n\
#include <stdlib.h>\n\
#include <iso646.h>\n\
#include <stdio.h>\n\
#include <string.h>\n\
#include <vector>\n\
#include <math.h>\n\
#include <assert.h>\n\
#include <stdarg.h>\n\
#include <iostream>\n\
#include <sstream>\n\
#include <fstream>\n\
using namespace std;\n\
int hlt_level;\n\
bool *hlt_exit;\n\
// Includes from nabla->parallel->includes()\n\%s",
          nabla->call->simd->includes(),
          nabla->call->parallel->includes());
  nMiddleDefines(nabla,nabla->call->header->defines);
  nMiddleTypedefs(nabla,nabla->call->header->typedefs);
  nMiddleForwards(nabla,nabla->call->header->forwards);
}


// ****************************************************************************
// * hookHeaderDump
// ****************************************************************************
void hookHeaderDump(nablaMain *nabla){
  assert(nabla->entity->name);
  dumpHeader(nabla);
}

// ****************************************************************************
// * hookHeaderPrefix
// ****************************************************************************
void hookHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name);
  fprintf(nabla->entity->hdr,
          "#ifndef __BACKEND_%s_H__\n#define __BACKEND_%s_H__",
          nabla->entity->name,
          nabla->entity->name);
}

// ****************************************************************************
// * ENUMERATES Hooks
// ****************************************************************************
void hookHeaderDefineEnumerates(nablaMain *nabla){
  const char *parallel_prefix_for_loop=nabla->call->parallel->loop(nabla);
  fprintf(nabla->entity->hdr,"\n\n\
/*********************************************************\n\
 * Forward enumerates\n\
 *********************************************************/\n\
#define FOR_EACH_PARTICLE(p) %sfor(int p=0;p<NABLA_NB_PARTICLES;p+=1)\n\
\n\
#define FOR_EACH_CELL(c) %sfor(int c=0;c<NABLA_NB_CELLS;c+=1)\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<NABLA_NODE_PER_CELL;n+=1)\n\
\n\
#define FOR_EACH_CELL_SHARED(c,local) %sfor(int c=0;c<NABLA_NB_CELLS;c+=1)\n\
\n\
/*#define FOR_EACH_CELL_NODE(n)\\\n\
  %sfor(int cn=c;cn>=c;--cn)\\\n\
    for(int n=NABLA_NODE_PER_CELL-1;n>=0;--n)\n\
*/\n\
#define FOR_EACH_NODE(n) /*prefix*/for(int n=0;n<NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE_CELL(c) for(int c=0,nc=NABLA_NODE_PER_CELL*n;c<NABLA_NODE_PER_CELL;c+=1,nc+=1)\n\
\n\
//#define FOR_EACH_NODE_CELL(c) for(int c=0;c<NABLA_NODE_PER_CELL;c+=1)\n\
\n\
#define FOR_EACH_FACE(f) /*prefix*/for(int f=0;f<NABLA_NB_FACES;f+=1)\n\
#define FOR_EACH_INNER_FACE(f) /*prefix*/for(int f=0;f<NABLA_NB_FACES_INNER;f+=1)\n\
#define FOR_EACH_OUTER_FACE(f) /*prefix*/for(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)\n\
// Pour l'instant un étant que multi-threadé, les 'own' sont les 'all'\n\
#define FOR_EACH_OWN_INNER_FACE(f) /*prefix*/for(int f=0;f<NABLA_NB_FACES_INNER;f+=1)\n\
#define FOR_EACH_OWN_OUTER_FACE(f) /*prefix*/for(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)\n\
",
          parallel_prefix_for_loop, // FOR_EACH_PARTICLE
          parallel_prefix_for_loop, // FOR_EACH_CELL
          parallel_prefix_for_loop, // FOR_EACH_NODE
          parallel_prefix_for_loop  // FOR_EACH_FACE
          );
}


/***************************************************************************** 
 * 
 *****************************************************************************/
void hookHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n#endif // __BACKEND_%s_H__\n",nabla->entity->name);
}
