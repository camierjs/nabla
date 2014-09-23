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
   variable->axl_it=true; // Par défaut, on dump la variable dans le fichier AXL
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
  // On cherche la primary_expression coté gauche du premier postfix_expression
  dbg("\n\t[nablaVariable] Looking for 'primary_expression':");
  astNode *nPrimaryExpression=dfsFetch(n->children,rulenameToId("primary_expression"));
  
  // Par contre, on va chercher les éventuels 'nabla_item' ou 'nabla_system' après le '['
  dbg("\n\t[nablaVariable] Looking for 'nabla_item':");
  astNode *nItem=dfsFetch(n->children->next->next,rulenameToId("nabla_item"));
  dbg("\n\t[nablaVariable] Looking for 'nabla_system':");
  astNode *nNablaSystem=dfsFetch(n->children->next->next,rulenameToId("nabla_system"));
  
  dbg("\n\t[nablaVariable] nPrimaryExpression->token=%s, nItem=%s, nNablaSystem=%s",
      (nPrimaryExpression!=NULL)?nPrimaryExpression->token:"NULL",
      (nItem!=NULL)?nItem->token:"NULL",
      (nNablaSystem!=NULL)?nNablaSystem->token:"NULL");
  
  if (nPrimaryExpression->token!=NULL && nPrimaryExpression!=NULL){
    nablaVariable *var=nablaVariableFind(nabla->variables, strdup(nPrimaryExpression->token));
    if (var!=NULL){
      // On a trouvé une variable nabla
      if (nItem==NULL && nNablaSystem==NULL){
        // Rien à faire, elle est bien postfixée mais par qq chose d'inconnu (variable locale?)
        nprintf(nabla, "/*no_item_system*/", NULL);
        return postfixed_nabla_variable_with_unknown;
      }
      if (nItem!=NULL && nNablaSystem==NULL){
        //fprintf(nabla->entity->src, "/*nablaVariable nItem*/m_%s_%s[", var->item, var->name);
        //nablaItem(nabla,cnf,nItem->token[0],enum_enum);
        nprintf(nabla, "/*is_item*/", NULL);
        return postfixed_nabla_variable_with_item;
      }
      if (nItem==NULL && nNablaSystem!=NULL){
        char *hookPrev=nabla->simd->prevCell();
        char *hookNext=nabla->simd->nextCell();
        char *hookNextPrev=(nNablaSystem->tokenid==NEXTCELL)?hookNext:hookPrev;
        nprintf(nabla, "/*is_system*/", "%s%s_%s",
                (nabla->backend==BACKEND_ARCANE)?"m_":hookNextPrev,
                var->item, var->name);
        nabla->hook->system(nNablaSystem,nabla,cnf,enum_enum);
        nprintf(nabla, "/*EndOf: is_system*/", "",NULL);
        // Variable postfixée par un mot clé system (prev/next, ...)
        return postfixed_nabla_system_keyword;
      }
    }else{
      nprintf(nabla, "/*var==NULL*/", NULL);
    }
  }else{
    nprintf(nabla, "/*token==NULL || nPrimaryExpression==NULL*/", NULL);
  }
  nprintf(nabla, "/*return postfixed_not_a_nabla_variable*/", NULL);
  return postfixed_not_a_nabla_variable;
}
