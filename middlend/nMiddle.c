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


// ****************************************************************************
// * Dump dans le header des 'includes'
// ****************************************************************************
NABLA_STATUS nMiddleCompoundJobEnd(nablaMain* nabla_main){
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump dans le header des 'includes'
 *****************************************************************************/
NABLA_STATUS nMiddleInclude(nablaMain *nabla, char *include){
  fprintf(nabla->entity->src, "%s\n", include);
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump dans le header les 'define's
 *****************************************************************************/
NABLA_STATUS nMiddleDefines(nablaMain *nabla, nablaDefine *defines){
  int i;
  FILE *target_file = isAnArcaneService(nabla)?nabla->entity->src:nabla->entity->hdr;
  fprintf(target_file,"\n\
\n// *****************************************************************************\
\n// * Defines\
\n// *****************************************************************************");
  for(i=0;defines[i].what!=NULL;i+=1)
    fprintf(target_file, "\n#define %s %s",defines[i].what,defines[i].with);
  fprintf(target_file, "\n");
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump dans le header les 'define's
 *****************************************************************************/
NABLA_STATUS nMiddleTypedefs(nablaMain *nabla, nablaTypedef *typedefs){
  fprintf(nabla->entity->hdr,"\n\
\n// *****************************************************************************\
\n// * Typedefs\
\n// *****************************************************************************");
  for(int i=0;typedefs[i].what!=NULL;i+=1)
    fprintf(nabla->entity->hdr, "\ntypedef %s %s;",typedefs[i].what,typedefs[i].with);
  fprintf(nabla->entity->hdr, "\n");
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump dans le header des 'forwards's
 *****************************************************************************/
NABLA_STATUS nMiddleForwards(nablaMain *nabla, char **forwards){
  fprintf(nabla->entity->hdr,"\n\
\n// *****************************************************************************\
\n// * Forwards\
\n// *****************************************************************************");
  for(int i=0;forwards[i]!=NULL;i+=1)
    fprintf(nabla->entity->hdr, "\n%s",forwards[i]);
  fprintf(nabla->entity->hdr, "\n");
  return NABLA_OK;
}


// ****************************************************************************
// * switchItemSupportTokenid
// ****************************************************************************
static char *switchItemSupportTokenid(int item_support_tokenid){
  if (item_support_tokenid==CELLS) return "cell";
  if (item_support_tokenid==FACES) return "face";
  if (item_support_tokenid==NODES) return "node";
  if (item_support_tokenid==PARTICLES) return "particle";
//#warning switchItemSupportTokenid should be deterministic
  return "global";
}


/*****************************************************************************
 * Fonction de parsing et d'application des actions correspondantes
 *****************************************************************************/
void nMiddleParseAndHook(astNode * n, nablaMain *nabla){
    
  ///////////////////////////////
  // Déclaration des libraries //
  ///////////////////////////////
  if (n->ruleid == rulenameToId("with_library")){
    dbg("\n\t[nablaMiddlendParseAndHook] with_library hit!");
    nMiddleLibraries(n,nabla->entity);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }
  
  ///////////////////////////////////////////////////////
  // Règle de définitions des items sur leurs supports //
  ///////////////////////////////////////////////////////
  if (n->ruleid == rulenameToId("nabla_item_definition")){
    char *item_support=n->children->children->token;
    // Nodes|Cells|Global|Faces|Particles
    int item_support_tokenid=n->children->children->tokenid;
    dbg("\n\t[nablaMiddlendParseAndHook] rule %s,  support %s", n->rule, item_support);
    // On backup temporairement le support (kind) de l'item
    nabla->tmpVarKinds=strdup(switchItemSupportTokenid(item_support_tokenid));
    nMiddleItems(n->children->next,
               rulenameToId("nabla_item_declaration"),
               nabla);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }

  ////////////////////////////////////
  // Règle de définitions des includes
  /////////////////////////////////////
  if (n->tokenid == INCLUDES){
    nMiddleInclude(nabla, n->token);
    dbg("\n\t[nablaMiddlendParseAndHook] rule hit INCLUDES %s", n->token);
  }

  ///////////////////////////////////
  // Règle de définitions des options
  ///////////////////////////////////
  if (n->ruleid == rulenameToId("nabla_options_definition")){
    dbg("\n\t[nablaMiddlendParseAndHook] rule hit %s", n->rule);
    nMiddleOptions(n->children,
                   rulenameToId("nabla_option_declaration"), nabla);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }

  /////////////////////////////////////////////////
  // On a une définition d'une fonction standard //
  /////////////////////////////////////////////////
  if (n->ruleid == rulenameToId("function_definition")){
    dbg("\n\t[nablaMiddlendParseAndHook] rule hit %s", n->rule);
    nabla->hook->function(nabla,n);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }
  
  ////////////////////////////////////////
  // On a une définition d'un job Nabla //
  ////////////////////////////////////////
  if (n->ruleid == rulenameToId("nabla_job_definition")){
    dbg("\n\t[nablaMiddlendParseAndHook] rule hit %s", n->rule);
    nabla->hook->job(nabla,n);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }

  //////////////////////////////
  // On a une reduction Nabla //
  //////////////////////////////
  if (n->ruleid == rulenameToId("nabla_reduction")){
    const astNode *global_node = n->children->next->next;
    const astNode *reduction_operation_node = global_node->next;
    const astNode *item_node = reduction_operation_node->next;
    char *global_var_name = global_node->token;
    char *item_var_name = item_node->token;
    dbg("\n\t[nablaMiddlendParseAndHook] rule hit %s", n->rule);
    dbg("\n\t[nablaMiddlendParseAndHook] Checking for global variable '%s'", global_var_name);
    const nablaVariable *global_var = nMiddleVariableFind(nabla->variables, global_var_name);
    const nablaVariable *item_var = nMiddleVariableFind(nabla->variables, item_var_name);
    dbg("\n\t[nablaMiddlendParseAndHook] global_var->item '%s'", global_var->item);
    // global_var must be 'global'
    assert(global_var->item[0]=='g');
    // item_var must not be 'global'
    assert(item_var->item[0]!='g');
    // Reduction operation is for now MIN
    assert(reduction_operation_node->tokenid==MIN_ASSIGN);
    // Having done these sanity checks, let's pass the rest of the generation to the backends
    nabla->hook->reduction(nabla,n);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }

  //////////////////
  // DFS standard //
  //////////////////
  if(n->children != NULL) nMiddleParseAndHook(n->children, nabla);
  if(n->next != NULL) nMiddleParseAndHook(n->next, nabla);
}


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


/// ***************************************************************************
// * nablaInsertSpace
// ****************************************************************************
void nMiddleInsertSpace( nablaMain *nabla, astNode * n){
  if (n->token!=NULL) {
    if (n->parent!=NULL){
      if (n->parent->rule!=NULL){
        if ( (n->parent->ruleid==rulenameToId("type_qualifier")) ||
             (n->parent->ruleid==rulenameToId("type_specifier")) ||
             (n->parent->ruleid==rulenameToId("jump_statement")) ||
             (n->parent->ruleid==rulenameToId("selection_statement")) ||
             (n->parent->ruleid==rulenameToId("storage_class_specifier"))
             ){
          nprintf(nabla, NULL, " ");
          //nprintf(nabla, NULL, "/*%s*/ ",n->parent->rule);
        }else{
          //nprintf(nabla, NULL, "/*%s*/",n->parent->rule);
        }
      }
    }
  }
}


// ****************************************************************************
// * nablaMiddlendSwitch
// ****************************************************************************
int nMiddleSwitch(astNode *root,
                  const bool optionDumpTree,
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
  nabla->simd=NULL;
  nabla->parallel=NULL;  
  dbg("\n\t[nablaMiddlendSwitch] On rajoute les variables globales");
  nMiddleVariableGlobalAdd(nabla);
  dbg("\n\t[nablaMiddlendSwitch] Now switching...");
  switch (backend){
  case BACKEND_ARCANE: return nccArcane(nabla,root,nabla_entity_name);
  case BACKEND_CUDA:   return nccCuda  (nabla,root,nabla_entity_name);
  case BACKEND_OKINA:  return nOkina (nabla,root,nabla_entity_name);
  default:
    exit(NABLA_ERROR
         |fprintf(stderr,
                  "\n[nablaMiddlendSwitch] Error while switching backend!\n"));
  }
  return NABLA_ERROR;
}
