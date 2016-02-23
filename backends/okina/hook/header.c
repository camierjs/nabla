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
#include "nabla.tab.h"

// ****************************************************************************
// * nOkinaHeader for Std, Avx or Mic
// ****************************************************************************
extern char knStdReal_h[];
extern char knStdReal3_h[];
extern char knStdInteger_h[];
extern char knStdGather_h[];
extern char knStdScatter_h[];
extern char knStdOStream_h[];
extern char knStdTernary_h[];
extern char knSseReal_h[];
extern char knSseReal3_h[];
extern char knSseInteger_h[];
extern char knSseGather_h[];
extern char knSseScatter_h[];
extern char knSseOStream_h[];
extern char knSseTernary_h[];
extern char knAvxReal_h[];
extern char knAvxReal3_h[];
extern char knAvxInteger_h[];
extern char knAvxGather_h[];
extern char knAvx2Gather_h[];
extern char knAvxScatter_h[];
extern char knAvxOStream_h[];
extern char knAvxTernary_h[];
extern char knMicReal_h[];
extern char knMicReal3_h[];
extern char knMicInteger_h[];
extern char knMicGather_h[];
extern char knMicScatter_h[];
extern char knMicOStream_h[];
extern char knMicTernary_h[];
static char *dumpExternalFile(char *file){
  return file+NABLA_LICENSE_HEADER;
}
static void nOkinaHeaderSimd(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  if ((nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC){
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicInteger_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicReal_h));
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicReal3_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicTernary_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicGather_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicScatter_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knMicOStream_h));
  }else if (((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)||
            ((nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)){
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxInteger_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxReal_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxReal3_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxTernary_h));
    if ((nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
      fprintf(nabla->entity->hdr,dumpExternalFile(knAvxGather_h));
    else
      fprintf(nabla->entity->hdr,dumpExternalFile(knAvx2Gather_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxScatter_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knAvxOStream_h));
  }else if ((nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE){
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseInteger_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseReal_h));
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseReal3_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseTernary_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseGather_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseScatter_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knSseOStream_h));
  }else{
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdInteger_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdReal_h));
    //if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdReal3_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdTernary_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdGather_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdScatter_h));
    fprintf(nabla->entity->hdr,dumpExternalFile(knStdOStream_h));
  }
}
// ****************************************************************************
// * nOkinaHeader for Dbg
// ****************************************************************************
extern char knDbg_h[];
static void nOkinaHeaderDbg(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(knDbg_h));
}
// ****************************************************************************
// * nOkinaHeader for Maths
// ****************************************************************************
extern char knMth_h[];
static void nOkinaHeaderMth(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(knMth_h));
}
// ****************************************************************************
// * nOkinaHeaderDump
// ****************************************************************************
void nOkinaHeaderDump(nablaMain *nabla){
  assert(nabla->entity->name);
  nOkinaHeaderSimd(nabla);
  nOkinaHeaderDbg(nabla);
  nOkinaHeaderMth(nabla);
}

// ****************************************************************************
// * nOkinaHeaderOpen
// ****************************************************************************
void nOkinaHeaderOpen(nablaMain *nabla){
  char hdrFileName[NABLA_MAX_FILE_NAME];
  // Ouverture du fichier header
  sprintf(hdrFileName, "%s.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
}


// ****************************************************************************
// * nOkinaEnumDefine
// ****************************************************************************
void nOkinaHeaderDefineEnumerates(nablaMain *nabla){
  const char *parallel_prefix_for_loop=nabla->call->parallel->loop(nabla);
  fprintf(nabla->entity->hdr,"\n\n\
/*********************************************************\n\
 * Forward enumerates\n\
 *********************************************************/\n\
#define FOR_EACH_CELL(c) %sfor(int c=0;c<NABLA_NB_CELLS;c+=1)\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<NABLA_NODE_PER_CELL;n+=1)\n\
\n\
#define FOR_EACH_CELL_WARP(c) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
#define FOR_EACH_CELL_WARP_SHARED(c,local) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
\n\
#define FOR_EACH_CELL_WARP_NODE(n)\\\n\
  %sfor(int cn=WARP_SIZE*c+WARP_SIZE-1;cn>=WARP_SIZE*c;--cn)\\\n\
    for(int n=NABLA_NODE_PER_CELL-1;n>=0;--n)\n\
\n\
#define FOR_EACH_NODE(n) /*%s*/for(int n=0;n<NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE_CELL(c) for(int c=0,nc=NABLA_NODE_PER_CELL*n;c<NABLA_NODE_PER_CELL;c+=1,nc+=1)\n\
\n\
#define FOR_EACH_NODE_WARP(n) %sfor(int n=0;n<NABLA_NB_NODES_WARP;n+=1)\n\
\n\
#define FOR_EACH_NODE_WARP_CELL(c)\\\n\
    for(int c=0;c<NABLA_NODE_PER_CELL;c+=1)\n",
          parallel_prefix_for_loop, // FOR_EACH_CELL
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_SHARED
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_NODE
          parallel_prefix_for_loop, // FOR_EACH_NODE
          parallel_prefix_for_loop  // FOR_EACH_NODE_WARP
          );
}


// ****************************************************************************
// * nOkinaHeaderPrefix
// ****************************************************************************
void nOkinaHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,
          "#ifndef __OKINA_%s_H__\n#define __OKINA_%s_H__",
          nabla->entity->name,nabla->entity->name);
}


// ****************************************************************************
// * nOkinaHeaderIncludes
// ****************************************************************************
void nOkinaHeaderIncludes(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,"\n\n\n\
// *****************************************************************************\n\
// * Okina includes\n\
// *****************************************************************************\n\
%s // from nabla->simd->includes\n\
#include <sys/time.h>\n\
#include <stdlib.h>\n\
#include <stdio.h>\n\
#include <string.h>\n\
#include <vector>\n\
#include <math.h>\n\
#include <assert.h>\n\
#include <stdarg.h>\n\
//#include <mathimf.h>\n\
#include <iostream>\n\
#include <sstream>\n\
#include <fstream>\n\
using namespace std;\n\
%s // fromnabla->parallel->includes()\n",
          nabla->call->simd->includes(),
          nabla->call->parallel->includes());
  nMiddleDefines(nabla,nabla->call->header->defines);
  nMiddleTypedefs(nabla,nabla->call->header->typedefs);
  nMiddleForwards(nabla,nabla->call->header->forwards);
}




// ****************************************************************************
// * okinaInclude
// ****************************************************************************
void nOkinaHeaderInclude(nablaMain *nabla){
  fprintf(nabla->entity->src,"#include \"%s.h\"\n", nabla->entity->name);
}



// ****************************************************************************
// * nOkinaHeaderPostfix
// ****************************************************************************
void nOkinaHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n#endif // __OKINA_%s_H__\n",nabla->entity->name);
}
