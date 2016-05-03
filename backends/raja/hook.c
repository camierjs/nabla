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

extern char raja_dump_h[];

// ****************************************************************************
// * kHookHeaderDump
// ****************************************************************************
void rajaHookHeaderDump(nablaMain *nabla){
  xHookHeaderDump(nabla);
  fprintf(nabla->entity->hdr,raja_dump_h+NABLA_LICENSE_HEADER);
}


// ****************************************************************************
// * rajaParallelIncludes
// ****************************************************************************
char *rajaParallelIncludes(void){
  return "\n//kParallelIncludes\n";
}


// ****************************************************************************
// * rajaHookHeaderIncludes
// ****************************************************************************
void rajaHookHeaderIncludes(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
#include <cmath>\n\
#include <cstdio>\n\
#include <cstdlib>\n\
#include <sstream>\n\
#include <fstream>\n\
#include <iostream>\n\
#include <cstring>\n\
#include <cctype>\n\
#include <assert.h>\n\
#include <sys/time.h>\n\
#include <RAJA/RAJA.hxx>\n\
#include <RAJA/IndexSetBuilders.hxx>\n\
using namespace std;\n\
");
  nMiddleDefines(nabla,nabla->call->header->defines);
  nMiddleTypedefs(nabla,nabla->call->header->typedefs);
  nMiddleForwards(nabla,nabla->call->header->forwards);
}


// ****************************************************************************
// * rajaHookHeaderPostfix
// ****************************************************************************
void rajaHookHeaderPostfix(nablaMain *nabla){
  xHookMeshStruct(nabla);
  fprintf(nabla->entity->hdr,"\n\n\
// *********************************************************\n\
// * Forward deep enumerates\n\
// *********************************************************\n\
#define FOR_EACH_NODE_MSH(n) for(int n=0;n<msh.NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE_CELL_MSH(c)\\\n\
\tfor(int c=0,nc=msh.NABLA_NODE_PER_CELL*n;c<msh.NABLA_NODE_PER_CELL;c+=1,nc+=1)\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<NABLA_NODE_PER_CELL;n+=1)\n\
#define FOR_EACH_NODE_CELL(c)\\\n\
\tfor(int c=0,nc=NABLA_NODE_PER_CELL*n;c<NABLA_NODE_PER_CELL;c+=1,nc+=1)");
  fprintf(nabla->entity->hdr,"\n\n#endif");
}


// ****************************************************************************
// * kHookEoe - End Of Enumerate
// ****************************************************************************
char* rajaHookEoe(nablaMain* nabla){
  return ");";
}


bool rajaHookDfsExtra(nablaMain* nabla,nablaJob* job,bool type){
  const char j=job->item[0];

  if (j=='c') nprintf(nabla, NULL,"%scellIdxSet,",type?"RAJA::IndexSet *":"");
  if (j=='n') nprintf(nabla, NULL,"%snodeIdxSet,",type?"RAJA::IndexSet *":"");
  
  return false;
}
