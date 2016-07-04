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
// * nMiddleInit
// ****************************************************************************
static nablaMain *nMiddleInit(const char *nabla_entity_name){
  nablaMain *nabla=(nablaMain*)calloc(1,sizeof(nablaMain));
  nablaEntity *entity; 
  nabla->name=strdup(nabla_entity_name);
  dbg("\n\t[nablaMiddlendInit] setting nabla->name to '%s'", nabla->name);
  dbg("\n\t[nablaMiddlendInit] Création de notre premier entity");
  entity=nMiddleEntityNew(nabla);
  dbg("\n\t[nablaMiddlendInit] Rajout du 'main'");
  nMiddleEntityAddEntity(nabla, entity);
  dbg("\n\t[nablaMiddlendInit] Rajout du nom de l'entity '%s'", nabla_entity_name);  
  entity->name=strdup(nabla_entity_name);
  entity->name_upcase=toolStrUpCase(nabla_entity_name);  // On lui rajoute son nom
  dbg("\n\t[nablaMiddlendInit] Rajout du name_upcase de l'entity %s", entity->name_upcase);  
  entity->main=nabla;                        // On l'ancre à l'unique entity pour l'instant
  assert(nabla->name != NULL);
  dbg("\n\t[nablaMiddlendInit] Returning nabla");
  return nabla;
}


// ****************************************************************************
// * nMiddleSwitch
// * The CUDA, KOKKOS & LAMBDA backends uses middlend/animate.c
// ****************************************************************************
NABLA_STATUS nMiddleSwitch(astNode *root,
                           const int optionDumpTree,
                           const char *nabla_entity_name,
                           const NABLA_BACKEND backend,
                           const BACKEND_OPTION option,
                           const BACKEND_PARALLELISM parallelism,
                           const BACKEND_COMPILER compiler,
                           char *interface_name,
                           char *specific_path,
                           char *service_name){
  nablaMain *nabla=nMiddleInit(nabla_entity_name);
  nabla->root=root;
  dbg("\n\t[nablaMiddlendSwitch] On initialise le type de backend\
 (= 0x%x) et de ses options (= 0x%x)",backend,option);
  nabla->backend=backend;
  nabla->option=option;
  nabla->parallelism=parallelism;
  nabla->compiler=compiler;
  nabla->interface_name=interface_name;
  nabla->specific_path=specific_path;
  nabla->service_name=service_name;
  nabla->optionDumpTree=optionDumpTree;
  nabla->options=NULL;  
  dbg("\n\t[nablaMiddlendSwitch] On rajoute les variables globales");
  middleGlobals(nabla);
  dbg("\n\t[nablaMiddlendSwitch] Now switching...");
  // Switching between our possible backends:
  switch (backend){
  case BACKEND_ARCANE: { nabla->hook=arcane(nabla); break;}
  case BACKEND_CUDA:   { nabla->hook=cuda(nabla); break;}
  case BACKEND_OKINA:  { nabla->hook=okina(nabla); break;}
  case BACKEND_LAMBDA: { nabla->hook=lambda(nabla); break;}
  case BACKEND_RAJA:   { nabla->hook=raja(nabla); break;}
  case BACKEND_KOKKOS: { nabla->hook=kokkos(nabla); break;}
  default:
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\nError while switching backend!\n"));
  }
  if (animate(nabla)!=NABLA_OK)
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\nError while animating backend!\n"));
  nMiddleOptionFree(nabla);
  nMiddleJobFree(nabla);
  free(nabla);
  return NABLA_OK;   
}
