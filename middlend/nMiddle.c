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
#include "nabla.tab.h"


/*****************************************************************************
 * Rajout des variables globales utiles aux mots clefs systèmes
 * On rajoute en dur les variables time, deltat, coord
 *****************************************************************************/
static void nMiddleVariableGlobalAdd(nablaMain *nabla){
  dbg("\n\t[nablaMiddlendVariableGlobalAdd] Adding global deltat, time");
  nablaVariable *deltat = nMiddleVariableNew(nabla);
  nMiddleVariableAdd(nabla, deltat);
  deltat->axl_it=false;
  deltat->item=strdup("global");
  deltat->type=strdup("real");
  deltat->name=strdup("deltat");
  nablaVariable *time = nMiddleVariableNew(nabla);
  nMiddleVariableAdd(nabla, time);
  time->axl_it=false;
  time->item=strdup("global");
  time->type=strdup("real");
  time->name=strdup("time");
  dbg("\n\t[nablaMiddlendVariableGlobalAdd] Adding AoS variables Real3 coord");
  nablaVariable *coord = nMiddleVariableNew(nabla);
  nMiddleVariableAdd(nabla, coord);
  coord->axl_it=true;
  coord->item=strdup("node");
  coord->type=strdup("real3");
  coord->name=strdup("coord");
  coord->field_name=strdup("NodeCoord");
}


/*****************************************************************************
 * nablaMiddlendInit
 *****************************************************************************/
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
// ****************************************************************************
int nMiddleSwitch(astNode *root,
                  const int optionDumpTree,
                  const char *nabla_entity_name,
                  const BACKEND_SWITCH backend,
                  const BACKEND_COLORS colors,
                  char *interface_name,
                  char *interface_path,
                  char *service_name){
  nablaMain *nabla=nMiddleInit(nabla_entity_name);
  dbg("\n\t[nablaMiddlendSwitch] On initialise le type de backend\
 (= 0x%x) et de ses variantes (= 0x%x)",backend,colors);
  nabla->backend=backend;
  nabla->colors=colors;
  nabla->interface_name=interface_name;
  nabla->interface_path=interface_path;
  nabla->service_name=service_name;
  nabla->optionDumpTree=optionDumpTree;
  nabla->options=NULL;  
  dbg("\n\t[nablaMiddlendSwitch] On rajoute les variables globales");
  nMiddleVariableGlobalAdd(nabla);
  dbg("\n\t[nablaMiddlendSwitch] Now switching...");
  // Switching between our possible backends:
  switch (backend){
  case BACKEND_ARCANE: return nccArcane(nabla,root,nabla_entity_name);
    // The CUDA backend now uses nMiddleBackendAnimate
    // Hook structures are filled by the backend    
  case BACKEND_CUDA:   {
    nabla->hook=cuda(nabla,root);
    return nMiddleBackendAnimate(nabla,root);
  }
  case BACKEND_OKINA:  return nOkina(nabla,root,nabla_entity_name);
    // The LAMBDA backend now uses nMiddleBackendAnimate
    // Hook structures are filled by the backend
  case BACKEND_LAMBDA: {
    nabla->hook=lambda(nabla);
    return nMiddleBackendAnimate(nabla,root);
  }
  default:
    exit(NABLA_ERROR|fprintf(stderr,
                  "\n[nablaMiddlendSwitch] Error while switching backend!\n"));
  }
  return NABLA_ERROR;
}

