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
#include "nabla.h"

extern nWhatWith arcaneOpCodesDefines[];

// ****************************************************************************
// * ifndef
// ****************************************************************************
static char* ifndef(char* path,const char* name){
  char* str=(char*)calloc(1024,sizeof(char));
  snprintf(str,1024,"_%s_%s_",
           toolStrUpCaseAndSwap(path,'/','_'),
           toolStrUpCaseAndSwap(name,'/','_'));
  return str;
}


// ****************************************************************************
// * aHookFamilyHeader
// ****************************************************************************
void aHookFamilyHeader(nablaMain *arc){
  const char *ifndef_token = ifndef(arc->specific_path,arc->name);
  const char *namespace_token = pth2nmspc(arc->specific_path);
  fprintf(arc->entity->hdr,"//aHookFamilyHeader\n\
#ifndef %s\n#define %s\n\n\
#include <arcane/IMesh.h>\n\
#include <arcane/VariableTypes.h>\n\
#include <arcane/IVariableAccessor.h>\n\
#include <arcane/VariableTypedef.h>\n\
#include <arcane/IParallelMng.h>\n\
#include <arcane/ISubDomain.h>\n\
#include <arcane/MeshAccessor.h>\n\
#include <arcane/utils/TraceAccessor.h>",
          ifndef_token,ifndef_token);
  
  nMiddleDefines(arc, arcaneOpCodesDefines);
  
  fprintf(arc->entity->hdr,"//#include <arcane/%s.h>\n\
\n\
using namespace Arcane;\n\
namespace %s{\n\
\tclass %s:\n\
\t\t//public %s,\n\
\t\tpublic TraceAccessor,\n\
\t\tpublic MeshAccessor{\n\
\tpublic:\n\
\t\t\t%s(IMesh *msh):TraceAccessor(msh->traceMng()),\n\
\t\t\t\t\t\t\t\t\tMeshAccessor(msh),\n\
\t\t\t\t\t\t\t\t\tm_sub_domain(msh->subDomain()){}\n\
\t\t\t~%s(){}\n\
\t\t\tISubDomain* m_sub_domain;\n\
\t\t\tISubDomain* subDomain() { return m_sub_domain; }\n\
\tpublic:\n",
          arc->interface_name,
          namespace_token,
          arc->name,
          arc->interface_name,
          arc->name,
          arc->name);
  fprintf(arc->entity->src,"//aHookFamilyHeader\n\
#include \"%s.h\"\nnamespace %s{\n",arc->name,namespace_token);
}


// ****************************************************************************
// * aHookFamilyVariablesPrefix
// ****************************************************************************
void aHookFamilyVariablesPrefix(nablaMain *arc){
  fprintf(arc->entity->hdr,"//aHookFamilyVariablesPrefix");
}


// ****************************************************************************
// * aHookFamilyFooter
// ****************************************************************************
void aHookFamilyFooter(nablaMain *arc){
  const char *ifndef_token = ifndef(arc->specific_path,arc->name);
  const char *namespace_token = pth2nmspc(arc->specific_path);
  fprintf(arc->entity->hdr,"\n//aHookFamilyFooter\n\t};\n\
} // namespace %s\n\
#endif // %s",
          namespace_token,
          ifndef_token);
  fprintf(arc->entity->src,"\n}//aHookFamilyFooter");
}
