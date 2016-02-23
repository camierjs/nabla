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

// ****************************************************************************
// * dumpExternalFile
// * NABLA_LICENSE_HEADER is tied and defined in nabla.h
// ****************************************************************************
static char *dumpExternalFile(char *file){
  return file+NABLA_LICENSE_HEADER;
}

extern char nablaAleph_h[];
extern char AlephStd_h[];
extern char AlephStd_c[];
extern char Aleph_h[];
extern char AlephTypesSolver_h[];
extern char AlephParams_h[];
extern char AlephVector_h[];
extern char AlephMatrix_h[];
extern char AlephKernel_h[];
extern char AlephOrdering_h[];
extern char AlephIndexing_h[];
extern char AlephTopology_h[];
extern char AlephInterface_h[];
extern char IAlephFactory_h[];

// *****************************************************************************
// * lambdaAlephIni
// *****************************************************************************
void lambdaAlephIni(nablaMain *main){
  nablaJob *alephIni=nMiddleJobNew(main->entity);
  alephIni->is_an_entry_point=true;
  alephIni->is_a_function=true;
  alephIni->scope  = strdup("NoScope");
  alephIni->region = strdup("NoRegion");
  alephIni->item   = strdup("\0");
  alephIni->return_type  = strdup("void");
  alephIni->name   = strdup("alephIni");
  alephIni->name_utf8 = strdup("ℵIni");
  alephIni->xyz    = strdup("NoXYZ");
  alephIni->direction  = strdup("NoDirection");
  sprintf(&alephIni->at[0],"-huge_valf");
  alephIni->when_index  = 1;
  alephIni->whens[0] = ENTRY_POINT_init;
  nMiddleJobAdd(main->entity, alephIni);  
}


// ****************************************************************************
// *
// ****************************************************************************
char* lambdaAlephHeader(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n");
  fprintf(nabla->entity->hdr,"/*'AlephStd_h'*/\n%s",dumpExternalFile(AlephStd_h));
  fprintf(nabla->entity->hdr,"/*'AlephStd_c'*/\n%s",dumpExternalFile(AlephStd_c));
  fprintf(nabla->entity->hdr,"/*'AlephTypesSolver_h'*/\n%s",dumpExternalFile(AlephTypesSolver_h));
  fprintf(nabla->entity->hdr,"/*'AlephParams_h'*/\n%s",dumpExternalFile(AlephParams_h));
  fprintf(nabla->entity->hdr,"/*'AlephVector_h'*/\n%s",dumpExternalFile(AlephVector_h));
  fprintf(nabla->entity->hdr,"/*'AlephMatrix_h'*/\n%s",dumpExternalFile(AlephMatrix_h));
  fprintf(nabla->entity->hdr,"/*'AlephKernel_h'*/\n%s",dumpExternalFile(AlephKernel_h));
  fprintf(nabla->entity->hdr,"/*'AlephOrdering_h'*/\n%s",dumpExternalFile(AlephOrdering_h));
  fprintf(nabla->entity->hdr,"/*'AlephIndexing_h'*/\n%s",dumpExternalFile(AlephIndexing_h));
  fprintf(nabla->entity->hdr,"/*'AlephTopology_h'*/\n%s",dumpExternalFile(AlephTopology_h));
  fprintf(nabla->entity->hdr,"/*'AlephInterface_h'*/\n%s",dumpExternalFile(AlephInterface_h));
  fprintf(nabla->entity->hdr,"/*'IAlephFactory_h'*/\n%s",dumpExternalFile(IAlephFactory_h));

  char str[NABLA_MAX_FILE_NAME];
  str[0]=0;
  // Et on rajoute les variables globales
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (strcmp(var->item, "global")!=0) continue;
    strcat(str, (var->type[0]=='r')?",real*":(var->type[0]=='i')?",int*":"/*Unknown type*/");
  }
  
  fprintf(nabla->entity->hdr, dumpExternalFile(nablaAleph_h));
  lambdaAlephIni(nabla);
  return "";
}


