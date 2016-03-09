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
   variable->in=false; 
   variable->out=false;
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


// *************************************************************
// * nMiddleVariableFind
// *************************************************************
nablaVariable *nMiddleVariableFind(nablaVariable *variables, char *name) {
  nablaVariable *variable=variables;
  assert(name!=NULL);
  //dbg("\n\t[nablaVariableFind] looking for '%s'", name);
  // Some backends use the fact it will return NULL
  //assert(variable != NULL);  assert(name != NULL);
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


// *************************************************************
// * nMiddleVariableFindWithJobItem
// *************************************************************
nablaVariable *nMiddleVariableFindWithSameJobItem(nablaMain *nabla,
                                                  nablaJob *job,
                                                  nablaVariable *variables,
                                                  const char *name){
  //const char cnfg=job->item[0];
  //const char item = job->item;
  nablaVariable *variable=variables;
  assert(name!=NULL);
  //if (job->is_a_function) return NULL;
  dbg("\n\t[nMiddleVariableFindWithJobItem] looking for '%s' of type '%s'", name, job->item);
  // Some backends use the fact it will return NULL
  //assert(variable != NULL);  assert(name != NULL);
  while(variable != NULL) {
    dbg(" ?%s:%s:%s", variable->name,variable->item, job->item);
    if ((strcmp(variable->name, name)==0) &&  // Au moins le même nom
                                              // ET au choix:
//#warning choix qui résume les non-homogénéités!
        ((strncmp(variable->item, job->item,4)==0)|| // exactement le même item
         (variable->item[0]=='g')||                  // une variable globale
         (strncmp(variable->name, "coord",5)==0)||   // coord
         (job->parse.alephKeepExpression==true)||    // aleph expression
         (job->parse.enum_enum!='\0')||              // une variable sous le foreach
         (job->is_a_function)||                      // synchronize par exemple
         (job->item[0]=='p')||                       // particle job
         (job->parse.isPostfixed!=0))){              // un accès à la boudarie condition 'pressure[0]'
      dbg(" Yes!");
      return variable;
    }
    variable = variable->next;
  }
  //dbg(" Nope!");
  //assert(variable!=NULL);
  return NULL;  
}


static void nMiddleVariablesSystemSwitch(nablaMain *nabla,
                                         int tokenid,
                                         char **prefix,
                                         char **system,
                                         char **postfix){
  switch(tokenid){
  case(NEXTCELL):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] NEXTCELL UNKNOWN");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,nextCell,DIR_UNKNOWN);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(NEXTCELL_X):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] NEXTCELL X");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,nextCell,DIR_X);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(NEXTCELL_Y):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] NEXTCELL Y");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,nextCell,DIR_Y);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(NEXTCELL_Z):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] NEXTCELL Z");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,nextCell,DIR_Z);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
    
  case(PREVCELL):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] PREVCELL UNKNOWN");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,prevCell,DIR_UNKNOWN);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(PREVCELL_X):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] PREVCELL X");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,prevCell,DIR_X);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(PREVCELL_Y):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] PREVCELL Y");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,prevCell,DIR_Y);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(PREVCELL_Z):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] PREVCELL Z");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,prevCell,DIR_Z);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(BACKCELL):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] BACKCELL");
    *prefix=cHOOKn(nabla,token,prefix);
    *system="";
    *postfix=cHOOKn(nabla,token,postfix);
    return;
  }
  case(FRONTCELL):{
    dbg("\n\t[nMiddleVariablesSystemSwitch] FRONTCELL");
    *prefix=cHOOKn(nabla,token,prefix);
    *system="";
    *postfix=cHOOKn(nabla,token,postfix);
    return;
  }
  default:return;
  }
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
  //dbg("\n\t[nMiddleVariables] Looking for 'primary_expression':");
  astNode *primary_expression=dfsFetch(n->children,rulenameToId("primary_expression"));
  // On va chercher l'éventuel 'nabla_item' après le '['
  //dbg("\n\t[nMiddleVariables] Looking for 'nabla_item':");
  astNode *nabla_item=dfsFetch(n->children->next->next,rulenameToId("nabla_item"));
  // On va chercher l'éventuel 'nabla_system' après le '['
  //dbg("\n\t[nMiddleVariables] Looking for 'nabla_system':");
  astNode *nabla_system=dfsFetch(n->children->next->next,rulenameToId("nabla_system"));
  
  /*dbg("\n\t[nMiddleVariables] primary_expression->token=%s, nabla_item=%s, nabla_system=%s",
      (primary_expression!=NULL)?primary_expression->token:"NULL",
      (nabla_item!=NULL)?nabla_item->token:"NULL",
      (nabla_system!=NULL)?nabla_system->token:"NULL");
  */
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
        dbg("\n\t[nMiddleVariables] Et elle est postfixée avec un nabla_system");
        char *prefix=NULL;
        char *system=NULL;
        char *postfix=NULL;
        nMiddleVariablesSystemSwitch(nabla,nabla_system->tokenid,&prefix,&system,&postfix);
        assert(prefix); assert(system); assert(postfix);
        dbg("\n\t[nMiddleVariables] prefix='%s'",prefix);
        dbg("\n\t[nMiddleVariables] system='%s'",system);
        dbg("\n\t[nMiddleVariables] postfix='%s'",postfix);
        nprintf(nabla, "/*is_system*/", "%s%s%s_%s%s",
                prefix,
                system,
                var->item, var->name,
                postfix);
        nabla->hook->token->system(nabla_system,nabla,cnf,enum_enum);
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


// ****************************************************************************
// * We check if the given variable name is used
// * in the current forall statement, given from the 'n' astNode
// ****************************************************************************
bool dfsUsedInThisForall(nablaMain *nabla, nablaJob *job, astNode *n,const char *name){
  nablaVariable *var=NULL;
  if (n==NULL){
    dbg("\n\t\t\t\t\t[dfsUsedInThisForall] NULL, returning");
    return false;
  }
  if (n->ruleid==rulenameToId("primary_expression")){
    dbg("\n\t\t\t\t\t[dfsUsedInThisForall] primary_expression");
    if (n->children->token!=NULL){
      dbg(", token: '%s'",n->children->token?n->children->token:"id");
      if ((var=nMiddleVariableFind(nabla->variables,n->children->token))!=NULL){
        dbg(", var-name: '%s' vs '%s'",var->name,name);
        if (strcmp(var->name,name)==0) return true;
      }
    }
  }
  if (n->children != NULL) if (dfsUsedInThisForall(nabla, job, n->children,name)) return true;
  if (n->next != NULL) if (dfsUsedInThisForall(nabla, job, n->next,name)) return true;
  return false;
}


// ****************************************************************************
// * dfsVariables
// * On scan pour lister les variables en in/inout/out
// ****************************************************************************
void dfsVariables(nablaMain *nabla, nablaJob *job, astNode *n,
                  bool left_of_assignment_expression){
  if (n==NULL) return;
  
  // Par défault, left_of_assignment_expression arrive à false
  // Si on tombe sur un assignment_expression, et un en fils en unary_expression
  // c'est qu'on passe à gauche du '=' et qu'on 'écrit'
  if (n->ruleid==rulenameToId("assignment_expression")&&
      (n->children->ruleid==rulenameToId("unary_expression"))){
    //dbg("\n\t\t\t[dfsVariables] left_of_assignment_expression @ TRUE!");
    left_of_assignment_expression=true;
  }
  // Si on passe par l'opérateur, on retombe du coté lecture
  if (n->ruleid==rulenameToId("assignment_operator")){
    //dbg("\n\t\t\t[dfsVariables] left_of_assignment_expression @ FALSE!");
    left_of_assignment_expression=false;
  }
  
  if (n->ruleid==rulenameToId("primary_expression")){
    if (n->children->tokenid == IDENTIFIER){
      //dbg("\n\t\t\t[dfsVariables] primary_expression!");
      const char *rw=(left_of_assignment_expression==true)?"WRITE":"READ";
      char* token = n->children->token;
      // Est-ce une variable connue?
      nablaVariable *var=nMiddleVariableFind(nabla->variables,token);
      //nablaVariable *var=nMiddleVariableFindWithSameJobItem(nabla,job,nabla->variables, token);
      if (var!=NULL){ // Si c'est bien un variable
        nablaVariable *new=nMiddleVariableFind(job->used_variables, token);
        //nablaVariable *new=nMiddleVariableFindWithSameJobItem(nabla,job,job->used_variables, token);
        if (new==NULL){ // Si c'est une variable qu'on a pas déjà placée dans la liste connue
          dbg("\n\t[dfsVariables] Variable '%s' is used (%s) in this job!",var->name,rw);
          // Création d'une nouvelle used_variable
          new = nMiddleVariableNew(NULL);
          new->name=strdup(var->name);
          new->item=var->item;//strdup((job->nb_in_item_set==0)?var->item:job->item);
          new->type=strdup(var->type);
          new->dim=var->dim;
          new->size=var->size;
          // Et on mémorise ce que l'on a fait en in/out
          if (left_of_assignment_expression) new->out|=true;
          if (!left_of_assignment_expression) new->in|=true;
          // Si elles n'ont pas le même support,
          // c'est qu'il va falloir insérer un gather/scatter
          dbg("\n\t[dfsVariables] new->item='%s' vs job->item='%s'",
              new->item, job->item);
          if (!job->is_a_function  && var->item[0]!='g' &&
              (new->item[0]!=job->item[0])){
            dbg("\n\t[dfsVariables] This variable will be gathered in this job!");
            //var->is_gathered=true;
            new->is_gathered=true;
         }
          // Rajout à notre liste
          if (job->used_variables==NULL)
            job->used_variables=new;
          else
            nMiddleVariableLast(job->used_variables)->next=new;
        }
        // Et on mémorise ce que l'on a fait en in/out
        if (left_of_assignment_expression)
          new->out|=true;
        else
          new->in|=true;
        //if (!left_of_assignment_expression) new->in|=true;
      }
      // Est-ce une option connue?
      const nablaOption *opt=nMiddleOptionFindName(nabla->options,token);
      if (opt!=NULL){
        nablaOption *new=nMiddleOptionFindName(job->used_options,token);
        if (new==NULL){
          assert(left_of_assignment_expression==false);
          dbg("\n\t[dfsVariables] Option '%s' is used (%s) in this job!",opt->name,rw);
          // Création d'une nouvelle used_option
          new = nMiddleOptionNew(nabla);
          new->axl_it=opt->axl_it;
          new->type=opt->type;
          new->name=opt->name;
          new->dflt=opt->dflt;
          new->main=nabla;        
          // Rajout à notre liste
          if (job->used_options==NULL)
            job->used_options=new;
          else
            nMiddleOptionLast(job->used_options)->next=new;
        }
      }
    }
  }
  if (n->children != NULL)
    dfsVariables(nabla, job, n->children,
                 left_of_assignment_expression);
  if (n->next != NULL)
    dfsVariables(nabla, job, n->next,
                 left_of_assignment_expression);
}


// ****************************************************************************
// * inout
// ****************************************************************************
static char* inout(const nablaVariable *var){
  if (var->in && var->out) return "inout";
  if (var->in && !var->out) return "in";
  if (!var->in && var->out) return "out";
  exit(NABLA_ERROR|fprintf(stderr, "\n[inout] in/inout/out error!\n"));
}


// ****************************************************************************
// * dfsVariablesDump
// ****************************************************************************
void dfsVariablesDump(nablaMain *nabla, nablaJob *job, astNode *n){
  nablaVariable *var=job->used_variables;
  dbg("\n\t[dfsVariablesDump]:");
  for(;var!=NULL;var=var->next){
    dbg("\n\t\t[dfsVariablesDump] Variable '%s' is used (%s) in this job!",var->name,inout(var));
  }
}
