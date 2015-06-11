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
 * Gestion des variables (items)
 *****************************************************************************/
nablaVariable *nMiddleVariableNew(nablaMain *arc){
	nablaVariable *variable;
	variable = (nablaVariable *)malloc(sizeof(nablaVariable));
 	assert(variable != NULL);
   variable->axl_it=true; // Par défaut, on dump la variable dans le fichier AXL
   variable->type=variable->name=variable->field_name=NULL;
   variable->main=arc;
   variable->next=NULL;
   variable->gmpRank=-1;
   variable->dump=true;
   //variable->used_in_a_forall=false;
   variable->is_gathered=false;
  	return variable; 
}


nablaVariable *nMiddleVariableAdd(nablaMain *arc, nablaVariable *variable) {
  assert(variable != NULL);
  if (arc->variables == NULL)
    arc->variables=variable;
  else
    nMiddleVariableLast(arc->variables)->next=variable;
  return NABLA_OK;
}


nablaVariable *nMiddleVariableLast(nablaVariable *variables) {
   while(variables->next != NULL){
     variables = variables->next;
   }
   return variables;
}

int nMiddleVariableGmpRank(nablaVariable *variables) {
  int rank=0;
  while(variables->next != NULL){
    if (variables->gmpRank!=-1) rank+=1;
    variables = variables->next;
  }
  return rank;
}
int nMiddleVariableGmpDumpNumber(nablaVariable *variables) {
  int number=0;
  while(variables->next != NULL){
    if (variables->gmpRank!=-1)
      if (variables->dump==true)
        number+=1;
    variables = variables->next;
  }
  return number;
}

char *nMiddleVariableGmpNameRank(nablaVariable *variables, int k) {
  int rank=-1;
  while(variables->next != NULL){
    if (variables->gmpRank!=-1) rank+=1;
    if (rank==k) return variables->name;
    variables = variables->next;
  }
  return NULL;
}

bool nMiddleVariableGmpDumpRank(nablaVariable *variables, int k) {
  int rank=-1;
  while(variables->next != NULL){
    if (variables->gmpRank!=-1) rank+=1;
    if (rank==k) return variables->dump;
    variables = variables->next;
  }
  return true;
}


/*
 * findTypeName
 */
nablaVariable *nMiddleVariableFind(nablaVariable *variables, char *name) {
  nablaVariable *variable=variables;
  dbg("\n\t[nablaVariableFind] looking for '%s'", name);
  // Some backends use the fact it will return NULL
  //assert(variable != NULL);  assert(name != NULL);
  while(variable != NULL) {
    dbg(" ?%s", variable->name);
    if(strcmp(variable->name, name) == 0){
      dbg(" Yes!");
      return variable;
    }
    variable = variable->next;
  }
  dbg(" Nope!");
  return NULL;
}



/*
 * 0 il faut continuer
 * 1 il faut returner
 * postfixed_nabla_variable_with_item ou  postfixed_nabla_variable_with_unknown il faut continuer en skippant le turnTokenToVariable
 */
what_to_do_with_the_postfix_expressions nMiddleVariables(nablaMain *nabla,
                                                         astNode * n,
                                                         const char cnf,
                                                         char enum_enum){
  // On cherche la primary_expression coté gauche du premier postfix_expression
  dbg("\n\t[nablaVariable] Looking for 'primary_expression':");
  astNode *primary_expression=dfsFetch(n->children,rulenameToId("primary_expression"));
  // On va chercher l'éventuel 'nabla_item' après le '['
  dbg("\n\t[nablaVariable] Looking for 'nabla_item':");
  astNode *nabla_item=dfsFetch(n->children->next->next,rulenameToId("nabla_item"));
  // On va chercher l'éventuel 'nabla_system' après le '['
  dbg("\n\t[nablaVariable] Looking for 'nabla_system':");
  astNode *nabla_system=dfsFetch(n->children->next->next,rulenameToId("nabla_system"));
  
  dbg("\n\t[nablaVariable] primary_expression->token=%s, nabla_item=%s, nabla_system=%s",
      (primary_expression!=NULL)?primary_expression->token:"NULL",
      (nabla_item!=NULL)?nabla_item->token:"NULL",
      (nabla_system!=NULL)?nabla_system->token:"NULL");
  // SI on a bien une primary_expression
  if (primary_expression->token!=NULL && primary_expression!=NULL){
    // On récupère de nom de la variable potentielle de cette expression
    nablaVariable *var=nMiddleVariableFind(nabla->variables, strdup(primary_expression->token));
    if (var!=NULL){ // On a bien trouvé une variable nabla
      if (nabla_item!=NULL && nabla_system!=NULL){
        // Mais elle est bien postfixée mais par qq chose d'inconnu: should be smarter here!
        // à-là: (node(0)==*this)?node(1):node(0), par exemple
        nprintf(nabla, "/*nabla_item && nabla_system*/", NULL);
        return postfixed_nabla_variable_with_unknown;
      }
      if (nabla_item==NULL && nabla_system==NULL){
        // Mais elle est bien postfixée mais par qq chose d'inconnu: variable locale?
        nprintf(nabla, "/*no_item_system*/", NULL);
        return postfixed_nabla_variable_with_unknown;
      }
      if (nabla_item!=NULL && nabla_system==NULL){
        // Et elle est postfixée avec un nabla_item
        //fprintf(nabla->entity->src, "/*nablaVariable nabla_item*/m_%s_%s[", var->item, var->name);
        //nablaItem(nabla,cnf,nabla_item->token[0],enum_enum);
        nprintf(nabla, "/*is_item*/", NULL);
        return postfixed_nabla_variable_with_item;
      }
      if (nabla_item==NULL && nabla_system!=NULL){
        // Et elle est postfixée avec un nabla_system
        char *hookPrev=nabla->simd->prevCell();
        char *hookNext=nabla->simd->nextCell();
        char *hookNextPrev=(nabla_system->tokenid==NEXTCELL)?hookNext:hookPrev;
        nprintf(nabla, "/*is_system*/", "%s%s_%s",
                (nabla->backend==BACKEND_ARCANE)?"m_":hookNextPrev,
                var->item, var->name);
        nabla->hook->system(nabla_system,nabla,cnf,enum_enum);
        nprintf(nabla, "/*EndOf: is_system*/", "",NULL);
        // Variable postfixée par un mot clé system (prev/next, ...)
        return postfixed_nabla_system_keyword;
      }
    }else{
      nprintf(nabla, "/*var==NULL*/", NULL);
    }
  }else{
    nprintf(nabla, "/*token==NULL || primary_expression==NULL*/", NULL);
  }
  nprintf(nabla, "/*return postfixed_not_a_nabla_variable*/", NULL);
  return postfixed_not_a_nabla_variable;
}
