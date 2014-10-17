/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nablaVariables.c     									       			  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2013.01.04          														  *
 * Updated  : 2013.01.04																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date		Author	Description					   									  *
 * 130104	camierjs	Creation				   											  *
 *****************************************************************************/
#include "nabla.h"
#include "nabla.tab.h"


/*****************************************************************************
 * Gestion des variables (items)
 *****************************************************************************/
nablaVariable *nablaVariableNew(nablaMain *arc){
	nablaVariable *variable;
	variable = (nablaVariable *)malloc(sizeof(nablaVariable));
 	assert(variable != NULL);
   variable->axl_it=true; // Par d�faut, on dump la variable dans le fichier AXL
   variable->type=variable->name=variable->field_name=NULL;
   variable->main=arc;
   variable->next=NULL;
   variable->gmpRank=-1;
   variable->dump=true;
   //variable->used_in_a_foreach=false;
   variable->is_gathered=false;
  	return variable; 
}


nablaVariable *nablaVariableAdd(nablaMain *arc, nablaVariable *variable) {
  assert(variable != NULL);
  if (arc->variables == NULL)
    arc->variables=variable;
  else
    nablaVariableLast(arc->variables)->next=variable;
  return NABLA_OK;
}


nablaVariable *nablaVariableLast(nablaVariable *variables) {
   while(variables->next != NULL){
     variables = variables->next;
   }
   return variables;
}

int nablaVariableGmpRank(nablaVariable *variables) {
  int rank=0;
  while(variables->next != NULL){
    if (variables->gmpRank!=-1) rank+=1;
    variables = variables->next;
  }
  return rank;
}
int nablaVariableGmpDumpNumber(nablaVariable *variables) {
  int number=0;
  while(variables->next != NULL){
    if (variables->gmpRank!=-1)
      if (variables->dump==true)
        number+=1;
    variables = variables->next;
  }
  return number;
}

char *nablaVariableGmpNameRank(nablaVariable *variables, int k) {
  int rank=-1;
  while(variables->next != NULL){
    if (variables->gmpRank!=-1) rank+=1;
    if (rank==k) return variables->name;
    variables = variables->next;
  }
  return NULL;
}

bool nablaVariableGmpDumpRank(nablaVariable *variables, int k) {
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
nablaVariable *nablaVariableFind(nablaVariable *variables, char *name) {
  nablaVariable *variable=variables;
  //dbg("\n\t[findVariableName] %s", name);
  assert(variable != NULL && name != NULL);
  while(variable != NULL) {
    //dbg(" ?%s", variable->name);
    if(strcmp(variable->name, name) == 0){
      //dbg(" Yes!");
      return variable;
    }
    variable = variable->next;
  }
  //dbg(" Nope!");
  return NULL;
}



/*
 * 0 il faut continuer
 * 1 il faut returner
 * postfixed_nabla_variable_with_item ou  postfixed_nabla_variable_with_unknown il faut continuer en skippant le turnTokenToVariable
 */
what_to_do_with_the_postfix_expressions nablaVariables(nablaMain *nabla,
                                                       astNode * n,
                                                       const char cnf,
                                                       char enum_enum){
  // On cherche la primary_expression cot� gauche du premier postfix_expression
  dbg("\n\t[nablaVariable] Looking for 'primary_expression':");
  astNode *primary_expression=dfsFetch(n->children,rulenameToId("primary_expression"));
  // On va chercher l'�ventuel 'nabla_item' apr�s le '['
  dbg("\n\t[nablaVariable] Looking for 'nabla_item':");
  astNode *nabla_item=dfsFetch(n->children->next->next,rulenameToId("nabla_item"));
  // On va chercher l'�ventuel 'nabla_system' apr�s le '['
  dbg("\n\t[nablaVariable] Looking for 'nabla_system':");
  astNode *nabla_system=dfsFetch(n->children->next->next,rulenameToId("nabla_system"));
  
  dbg("\n\t[nablaVariable] primary_expression->token=%s, nabla_item=%s, nabla_system=%s",
      (primary_expression!=NULL)?primary_expression->token:"NULL",
      (nabla_item!=NULL)?nabla_item->token:"NULL",
      (nabla_system!=NULL)?nabla_system->token:"NULL");
  // SI on a bien une primary_expression
  if (primary_expression->token!=NULL && primary_expression!=NULL){
    // On r�cup�re de nom de la variable potentielle de cette expression
    nablaVariable *var=nablaVariableFind(nabla->variables, strdup(primary_expression->token));
    if (var!=NULL){ // On a bien trouv� une variable nabla
      if (nabla_item!=NULL && nabla_system!=NULL){
        // Mais elle est bien postfix�e mais par qq chose d'inconnu: should be smarter here!
        // �-l�: (node(0)==*this)?node(1):node(0), par exemple
        nprintf(nabla, "/*nabla_item && nabla_system*/", NULL);
        return postfixed_nabla_variable_with_unknown;
      }
      if (nabla_item==NULL && nabla_system==NULL){
        // Mais elle est bien postfix�e mais par qq chose d'inconnu: variable locale?
        nprintf(nabla, "/*no_item_system*/", NULL);
        return postfixed_nabla_variable_with_unknown;
      }
      if (nabla_item!=NULL && nabla_system==NULL){
        // Et elle est postfix�e avec un nabla_item
        //fprintf(nabla->entity->src, "/*nablaVariable nabla_item*/m_%s_%s[", var->item, var->name);
        //nablaItem(nabla,cnf,nabla_item->token[0],enum_enum);
        nprintf(nabla, "/*is_item*/", NULL);
        return postfixed_nabla_variable_with_item;
      }
      if (nabla_item==NULL && nabla_system!=NULL){
        // Et elle est postfix�e avec un nabla_system
        char *hookPrev=nabla->simd->prevCell();
        char *hookNext=nabla->simd->nextCell();
        char *hookNextPrev=(nabla_system->tokenid==NEXTCELL)?hookNext:hookPrev;
        nprintf(nabla, "/*is_system*/", "%s%s_%s",
                (nabla->backend==BACKEND_ARCANE)?"m_":hookNextPrev,
                var->item, var->name);
        nabla->hook->system(nabla_system,nabla,cnf,enum_enum);
        nprintf(nabla, "/*EndOf: is_system*/", "",NULL);
        // Variable postfix�e par un mot cl� system (prev/next, ...)
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
