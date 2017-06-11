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
#include "nabla.tab.h"

// ****************************************************************************
// * nablaInsertSpace
// ****************************************************************************
void nMiddleInsertSpace(nablaMain *nabla, node * n){
  if (n->token!=NULL) {
    if (n->parent!=NULL){
      if (n->parent->rule!=NULL){
        if ((n->parent->ruleid==ruleToId(rule_type_qualifier)) ||
            (n->parent->ruleid==ruleToId(rule_type_specifier)) ||
            (n->parent->ruleid==ruleToId(rule_jump_statement)) ||
            (n->parent->ruleid==ruleToId(rule_selection_statement)) ||
            (n->parent->ruleid==ruleToId(rule_storage_class_specifier))){
          nprintf(nabla, NULL, " ");
          //nprintf(nabla, NULL, "/*%s*/ ",n->parent->rule);
        }
      }
    }
  }
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


// ****************************************************************************
// * nMiddleDeclarationDump
// ****************************************************************************
static void nMiddleDeclarationDump(nablaMain *nabla, node * n){
  for(;n->token != NULL;){
    dbg(" %s",n->token);
    nprintf(nabla, NULL, "%s ", n->token);
    if (n->tokenid == ';'){
      //nprintf(nabla, NULL, "\n");
      return;
    }
    break;
  }
  if(n->children != NULL) nMiddleDeclarationDump(nabla, n->children);
  if(n->next != NULL) nMiddleDeclarationDump(nabla, n->next);
}


// ****************************************************************************
// * Fonction de parsing et d'application des actions correspondantes
// * On scrute les possibilités du 'nabla_grammar':
// *    - INCLUDES
// *    - preproc
// *    - with_library
// *    - declaration
// *    - nabla_options_definition
// *    - nabla_item_definition
// *    - function_definition
// *    - nabla_job_definition
// *    - nabla_reduction
// ****************************************************************************
void nMiddleGrammar(node *n, nablaMain *nabla){
    
  /////////////////////////////////////
  // Règle de définitions des includes
  /////////////////////////////////////
  if (n->tokenid == INCLUDES){
    nMiddleInclude(nabla, n->token);
    dbg("\n*** [nMiddleGrammar] rule hit INCLUDES %s", n->token);
  }

  /////////////////////////////////////
  // Règle de définitions des preprocs 
  /////////////////////////////////////
  if (n->ruleid == ruleToId(rule_preproc)){
    // We do not want preprocs declarations down to the source file
    //nprintf(nabla, NULL, "%s\n", n->children->token);
    dbg("\n*** [nMiddleGrammar] preproc '%s'", n->children->token);
  }
  
  ///////////////////////////////
  // Déclaration des libraries //
  ///////////////////////////////
  if (n->ruleid == ruleToId(rule_with_library)){
    dbg("\n*** [nMiddleGrammar] with_library hit!");
    nMiddleLibraries(n,nabla->entity);
    dbg("\n\t[nMiddleGrammar] library done");
    dbg("\n\t[nMiddleGrammar] nabla->entity->libraries=0x%X",
        nabla->entity->libraries);
  }

  //////////////////////////////
  // Règle ∇ de 'declaration' //
  //////////////////////////////
  if (n->ruleid == ruleToId(rule_declaration)){
    dbg("\n*** [nMiddleGrammar] declaration hit:");
    nMiddleDeclarationDump(nabla,n);
    nprintf(nabla, NULL, "\n");
    dbg(", done");
  }
  
  ///////////////////////////////////////////////////////
  // Règle de définitions des items sur leurs supports //
  ///////////////////////////////////////////////////////
  if (n->ruleid == ruleToId(rule_nabla_item_definition)){
    const char *item_support=n->children->children->token;
    // Nodes|Cells|Global|Faces|Particles
    int item_support_tokenid=n->children->children->tokenid;
    dbg("\n*** [nMiddleGrammar] rule %s on '%s'",
        n->rule, item_support);
    // On backup temporairement le support (kind) de l'item
    nabla->tmpVarKinds=sdup(switchItemSupportTokenid(item_support_tokenid));
    nMiddleItems(n->children->next,
               ruleToId(rule_nabla_item_declaration),
               nabla);
    dbg("\n\t[nMiddleGrammar] item done");
  }

  //////////////////////////////////////
  // Règle de définitions des options //
  //////////////////////////////////////
  if (n->ruleid == ruleToId(rule_nabla_options_definition)){
    dbg("\n*** [nMiddleGrammar] rule hit %s", n->rule);
    nMiddleOptions(n->children,
                   ruleToId(rule_nabla_option_declaration), nabla);
    dbg("\n\t[nMiddleGrammar] option done");
  }

  /////////////////////////////////////////////////
  // On a une définition d'une fonction standard //
  /////////////////////////////////////////////////
  if (n->ruleid == ruleToId(rule_function_definition)){
    dbg("\n*** [nMiddleGrammar] rule hit %s", n->rule);
    bool is_an_entry_point=false;
    // On scan pour trouver un '@'
    for(node *nAt=n->children;nAt not_eq NULL;nAt=nAt->next){
      //if (nAt->token!=NULL) dbg("%s ",nAt->token);
      if (nAt->tokenid!=AT) continue;
      is_an_entry_point=true;
      break;
    }
    if (nabla->hook->grammar->hit &&
        !nabla->hook->grammar->hit(nabla,is_an_entry_point))
      goto step_next;
    
    nablaJob *fct = nMiddleJobNew(nabla->entity);
    nMiddleJobAdd(nabla->entity, fct);
    nMiddleFunctionFill(nabla,fct,n,
                        nabla->hook->source->name?
                        nabla->hook->source->name(nabla):NULL);
    dbg("\n\t[nMiddleGrammar] function done");
    goto step_next;
  }
  
  ////////////////////////////////////////
  // On a une définition d'un job Nabla //
  ////////////////////////////////////////
  if (n->ruleid == ruleToId(rule_nabla_job_definition)){
    dbg("\n*** [nMiddleGrammar] rule hit %s", n->rule);
    bool job_to_add=true;
    bool is_an_entry_point=false;
    
    // On scan pour trouver un '@'
    for(node *nAt=n->children;nAt not_eq NULL;nAt=nAt->next){
      //if (nAt->token!=NULL) dbg("%s ",nAt->token);
      if (nAt->tokenid!=AT) continue;
      is_an_entry_point=true;
      break;
    }
    dbg("\n\t\t[nMiddleGrammar] is_an_entry_point=%s",
        is_an_entry_point?"true":"false");
    if (nabla->hook->grammar->hit &&
        !nabla->hook->grammar->hit(nabla,is_an_entry_point))
      job_to_add=false;
      
    if (!job_to_add){
      dbg("\n\t[nMiddleGrammar] Skipping job!");
      goto step_next;
    }else{
      dbg("\n\t[nMiddleGrammar] nMiddleJobNew!");
      nablaJob *job = nMiddleJobNew(nabla->entity);
      dbg("\n\t[nMiddleGrammar] nMiddleJobAdd!");
      nMiddleJobAdd(nabla->entity, job);
      dbg("\n\t[nMiddleGrammar] nMiddleJobFill!");
      nMiddleJobFill(nabla,job,n,
                     nabla->hook->source->name?
                     nabla->hook->source->name(nabla):NULL);
      dbg("\n\t[nMiddleGrammar] job done");
    }
    goto step_next;
  }

  //////////////////////////////
  // On a une reduction Nabla //
  //////////////////////////////
  if (n->ruleid == ruleToId(rule_nabla_reduction)){
    node *global_node = n->children->next;
    node *item_node =n->children->next->next->next;
    const char *global_var_name = global_node->token;
    const char *item_var_name = item_node->token;
    dbg("\n*** [nMiddleGrammar] rule hit %s", n->rule);
    dbg("\n\t[nMiddleGrammar] Checking for global variable '%s'",
        global_var_name);
    const nablaVariable *global_var =
      nMiddleVariableFind(nabla->variables, global_var_name);
    const nablaVariable *item_var =
      nMiddleVariableFind(nabla->variables, item_var_name);
    dbg("\n\t[nMiddleGrammar] global_var->item '%s'", global_var->item);
    dbg("\n\t[nMiddleGrammar] item_var->item '%s'", item_var->item);
    // global_var must be 'global'
    assert(global_var->item[0]=='g');
    // item_var must not be 'global'
    assert(item_var->item[0]!='g');
    // Having done these sanity checks,
    // let's pass the rest of the generation to the backends
    dbg("\n\t[nMiddleGrammar] Hooking reduction:");
    nabla->hook->grammar->reduction(nabla,n);
    dbg("\n\t[nMiddleGrammar] reduction done");
  }
  
  //////////////////
  // DFS standard //
  //////////////////
  if(n->children != NULL) nMiddleGrammar(n->children, nabla);
 step_next:
  if(n->next != NULL) nMiddleGrammar(n->next, nabla);
}
