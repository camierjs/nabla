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

void nMiddleVariableFree(nablaVariable *variable){
  for(nablaVariable *this,*var=this=variable;var!=NULL;free(this)){
    //if (this->power_type) free(this->power_type);
    var=(this=var)->next;
  }
}

// ****************************************************************************
// * nMiddleVariableNew: New, Last, Add & Find
// ****************************************************************************
nablaVariable *nMiddleVariableNew(nablaMain *nabla){
	nablaVariable *variable;
	variable = (nablaVariable *)calloc(1,sizeof(nablaVariable));
 	assert(variable);
   variable->item=NULL;
   variable->type=NULL;
   variable->name=NULL;
   variable->dim=0;
   variable->size=0;
   variable->koffset=0;
   variable->vitem=0;
   variable->dump=true;
   variable->in=false; 
   variable->out=false;
   variable->is_gathered=false;
   variable->inout=enum_in_variable;
   variable->axl_it=true;
   variable->gmpRank=-1;
   variable->power_type=NULL;
   variable->main=nabla;
   variable->next=NULL;
  	return variable; 
}

nablaVariable *nMiddleVariableAdd(nablaMain *nabla, nablaVariable *variable) {
  assert(variable != NULL);
  if (nabla->variables == NULL)
    nabla->variables=variable;
  else
    nMiddleVariableLast(nabla->variables)->next=variable;
  return NABLA_OK;
}

nablaVariable *nMiddleVariableLast(nablaVariable *variables) {
   while(variables->next != NULL)
     variables = variables->next;
   return variables;
}

nablaVariable *nMiddleVariableFindKoffset(nablaVariable *variables,
                                          const char *name,
                                          const int koffset){
  nablaVariable *variable=variables;
  assert(name!=NULL);
  dbg("\n\t\t\t\t\t\t[nablaVariableFind] looking for '%s-k(%d)'", name,koffset);
  while(variable) {
    dbg(" ?%s-k(%d)", variable->name,variable->koffset);
    if (strcmp(variable->name, name)==0  // Bon name
        && variable->koffset==koffset){  // Bon k-offset 
      dbg(" Yes!");
      return variable;
    }
    variable = variable->next;
  }
  dbg(" Nope!");
  return NULL;
}

nablaVariable *nMiddleVariableFind(nablaVariable *variables, const char *name) {
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

// ****************************************************************************
// * nMiddleVariableGmp: Rank, DumpNumber, NameRank, DumpRank
// ****************************************************************************
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
                                         nablaJob *job,
                                         int tokenid,
                                         char **prefix,
                                         char **system,
                                         char **postfix){
  // We want a job and not a function
  assert(job->item!=NULL);
  assert(!job->is_a_function);
  // If the job is using the '_XYZ' functionnality, use it
  dbg("\n\t[nMiddleVariablesSystemSwitch] xyz=%s, direction=%s",job->xyz,job->direction);
  dbg("\n\t[nMiddleVariablesSystemSwitch] tokenid=%d",tokenid);
  const int dir = (job->xyz!=NULL)?
    (job->direction[6]=='X')?DIR_X:
    (job->direction[6]=='Y')?DIR_Y:
    (job->direction[6]=='Z')?DIR_Z:DIR_UNKNOWN:
    DIR_UNKNOWN;
  const char itm = job->item[0];
  switch(tokenid){
    
  case(ARROW_UP):{ // ⇒ next in Y direction
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_UP\n");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,nextCell,DIR_Y);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(ARROW_RIGHT):{ // ⇒ next in X direction
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_RIGHT\n");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,nextCell,DIR_X);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(ARROW_DOWN):{ // ⇒ prev in Y direction
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_DOWN\n");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,prevCell,DIR_Y);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(ARROW_LEFT):{ // ⇒ prev in X direction
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_LEFT\n");
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,prevCell,DIR_X);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }

  case(ARROW_NORTH_EAST):{ // ⇒ ↗
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_NORTH_EAST\n");
    *prefix="ARROW";
    *system="NORTH";
    *postfix="EAST";
    return;
  }
  case(ARROW_SOUTH_EAST):{ // ⇒ ↘
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_SOUTH_EAST\n");
    *prefix="ARROW";
    *system="SOUTH";
    *postfix="EAST";
    return;
  }
  case(ARROW_SOUTH_WEST):{ // ⇒ ↙
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_SOUTH_WEST\n");
    *prefix="ARROW";
    *system="SOUTH";
    *postfix="WEST";
    return;
  }
  case(ARROW_NORTH_WEST):{ // ⇒ ↖
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_NORTH_WEST\n");
    *prefix="ARROW";
    *system="NORTH";
    *postfix="WEST";
    return;
  }

  case(ARROW_BACK):{ // ⇒ ⊠
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_BACK\n");
    *prefix="ARROW";
    *system="_";
    *postfix="BACK";
    return;
  }
  case(ARROW_FRONT):{ // ⇒ ⊡
    dbg("\n\t[nMiddleVariablesSystemSwitch] ARROW_FRONT\n");
    *prefix="ARROW";
    *system="_";
    *postfix="FRONT";
    return;
  }
    
  case(NEXT):{
    assert(itm=='c');// For now, we just work with cell jobs
    dbg("\n\t[nMiddleVariablesSystemSwitch] NEXT prefix (dir=%d, DIR_X=%d, direction=%s)",dir,DIR_X,job->direction);
    *prefix=cHOOK(nabla,xyz,prefix);
    dbg("\n\t[nMiddleVariablesSystemSwitch] NEXT system");
    *system=cHOOKi(nabla,xyz,nextCell,dir);
    dbg("\n\t[nMiddleVariablesSystemSwitch] NEXT postfix");
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
  case(PREV):{
    assert(itm=='c');// For now, we just work with cell jobs
    dbg("\n\t[nMiddleVariablesSystemSwitch] PREV %d ?",dir);
    *prefix=cHOOK(nabla,xyz,prefix);
    *system=cHOOKi(nabla,xyz,prevCell,dir);
    *postfix=cHOOK(nabla,xyz,postfix);
    return;
  }
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

// ****************************************************************************
// * 0 il faut continuer
// * 1 il faut returner
// * postfixed_nabla_variable_with_item
// * postfixed_nabla_variable_with_unknown
// *    postfixed_* ⇒ il faut continuer en skippant le turnTokenToVariable
// ****************************************************************************
// * C'est une des fonctions qu'il faudrait revoir et même supprimer
// ****************************************************************************
what_to_do_with_the_postfix_expressions nMiddleVariables(nablaMain *nabla,
                                                         nablaJob *job,
                                                         node * n,
                                                         const char cnf,
                                                         char enum_enum){
  // On cherche la primary_expression coté gauche du premier postfix_expression
  //dbg("\n\t[nMiddleVariables] Looking for 'primary_expression':");
  node *primary_expression=dfsFetch(n->children,ruleToId(rule_primary_expression));
  // On va chercher l'éventuel 'nabla_item' après le '['
  //dbg("\n\t[nMiddleVariables] Looking for 'nabla_item':");
  node *nabla_item=dfsFetch(n->children->next->next,ruleToId(rule_nabla_item));
  // On va chercher l'éventuel 'nabla_system' après le '['
  //dbg("\n\t[nMiddleVariables] Looking for 'nabla_system':");
  node *nabla_system=dfsFetch(n->children->next->next,ruleToId(rule_nabla_system));
  
  /*dbg("\n\t[nMiddleVariables] primary_expression->token=%s, nabla_item=%s, nabla_system=%s",
      (primary_expression!=NULL)?primary_expression->token:"NULL",
      (nabla_item!=NULL)?nabla_item->token:"NULL",
      (nabla_system!=NULL)?nabla_system->token:"NULL");
  */
  // SI on a bien une primary_expression
  if (primary_expression->token!=NULL && primary_expression!=NULL){
    // On récupère de nom de la variable potentielle de cette expression
    nablaVariable *var=nMiddleVariableFind(nabla->variables, sdup(primary_expression->token));
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
        nMiddleVariablesSystemSwitch(nabla,job,nabla_system->tokenid,&prefix,&system,&postfix);
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
// * [#] ⇒ 0, [#+[0-9]*] ⇒ [0-9]*
// ****************************************************************************
static int getKoffset(char *k){
  assert(k[0]=='[');
  assert(k[1]=='#');
  const int i=atoi(&k[2]);
  //dbg("\n\t\t\t\t[getKoffset] k='%s', i=%d!",k,i);
  return i;
}


// ****************************************************************************
// * We check if the given variable name is used
// * in the current forall statement, from the given 'n' node
// ****************************************************************************
bool dfsUsedInThisForallKoffset(nablaMain *nabla,
                                nablaJob *job,
                                node *n,
                                const char *name,
                                const int koffset){
  nablaVariable *var=NULL;
  if (n==NULL){
    dbg("\n\t\t\t\t\t[dfsUsedInThisForallKoffset] NULL, returning");
    return false;
  }
  if (n->ruleid==ruleToId(rule_primary_expression)){
    dbg("\n\t\t\t\t\t[dfsUsedInThisForallKoffset] primary_expression (%s)", n->children->token?n->children->token:"?");
    if ((n->children->tokenid==IDENTIFIER) && (strcmp(n->children->token,name)==0)){
      const bool id_has_a_koffset = (n->children->next &&
                                     n->children->next->tokenid==K_OFFSET);
      const int n_children_next_koffset=id_has_a_koffset?getKoffset(n->children->next->token):0;
      dbg(", token: '%s-k(%d) vs arg:%s-k(%d)",
          n->children->token, n_children_next_koffset,
          name,koffset);
      if ((var=nMiddleVariableFindKoffset(job->used_variables,//nabla->variables,
                                          n->children->token,
                                          n_children_next_koffset))!=NULL){
        dbg(", hit var-name-k: '%s' vs '%s'",var->name,name);
        if (strcmp(var->name,name)==0){
          dbg("YES");
          return true;
        }
      }
    }
  }
  if (n->children != NULL) if (dfsUsedInThisForallKoffset(nabla, job, n->children,name,koffset)) return true;
  if (n->next != NULL) if (dfsUsedInThisForallKoffset(nabla, job, n->next,name,koffset)) return true;
  return false;
}
bool dfsUsedInThisForall(nablaMain *nabla, nablaJob *job, node *n,const char *name){
  //dbg("\n\t\t\t\t\t[dfsUsedInThisForall]");
  return dfsUsedInThisForallKoffset(nabla,job,n,name,0);
}

// ****************************************************************************
// * dfsAddToUsedVariables
// ****************************************************************************
static void dfsAddToUsedVariables(nablaJob *job,
                                  const char *token,
                                  nablaVariable *var,
                                  const int koffset,
                                  const char *rw,
                                  const bool left_of_assignment_expression){
  // Ci c'est déjà une variable connue, on a rien à faire
  if (nMiddleVariableFindKoffset(job->used_variables,token,koffset)){
    dbg("\n\t\t\t[dfsAddToUsedVariables] Variable '%s-%d' is allready used (%s) in this job!",var->name,koffset,rw);
    return;
  }
  // Si ce n'est pas une variable déjà placée
  // dans la liste des utilisées, on la rajoute
  dbg("\n\t\t\t[dfsAddToUsedVariables] Variable '%s-%d' is used (%s) in this job!",var->name,koffset,rw);
  // Création de cette nouvelle used_variable
  nablaVariable *new=nMiddleVariableNew(NULL);
  new->name=sdup(var->name);
  new->item=var->item;
  new->type=sdup(var->type);
  new->dim=var->dim;
  new->vitem=var->vitem;
  new->size=var->size;
  new->koffset=koffset;
  // Et on mémorise ce que l'on a fait en in/out
  if (left_of_assignment_expression) new->out|=true;
  if (!left_of_assignment_expression) new->in|=true;
  // Si elles n'ont pas le même support,
  // c'est qu'il va falloir insérer un gather/scatter
  dbg("\n\t\t\t[dfsAddToUsedVariables] new->name=%s new->item='%s' vs job->item='%s'",
      new->name, new->item, job->item);
  if (!job->is_a_function && // que pour les jobs
      var->item[0]!='g' && // pas de gather des globales
      var->item[0]!='x' && // pas de gather des xs
      new->item[0]!=job->item[0]){
    dbg("\n\t\t\t[dfsAddToUsedVariables] This variable will be gathered in this job!");
    new->is_gathered=true;
  }
  // Rajout à notre liste
  if (job->used_variables==NULL) job->used_variables=new;
  else nMiddleVariableLast(job->used_variables)->next=new;
  // Et on mémorise ce que l'on a fait en in/out
  if (left_of_assignment_expression) new->out|=true;
  else new->in|=true;
}

// ****************************************************************************
// * dfsVarThisOneSpecial
// ****************************************************************************
nablaVariable* dfsVarThisOneSpecial(char* token, char* test, int d){
  if (strcmp(token,test)!=0) return NULL;
  nablaVariable *var=nMiddleVariableNew(NULL);
  var->name=sdup(token);
  var->item=sdup("xs");
  var->type=sdup("int");
  var->dim=d;
  return var;
}

// ****************************************************************************
// * dfsVarThisOne
// ****************************************************************************
static void dfsVarThisOne(nablaMain *nabla,
                          nablaJob *job,
                          node *n,
                          const int tokenid,
                          char* token,
                          const bool left_of_assignment_expression){
  if (n->children->tokenid != tokenid) return;
  dbg("\n\t\t\t\t[dfsVarThisOne] nabla_system (%s, id #%d)!", token,tokenid);
  const char *rw=(left_of_assignment_expression==true)?"WRITE":"READ";
  nablaVariable *var=nMiddleVariableFind(nabla->variables,token);
  if (var==NULL) var=dfsVarThisOneSpecial(token,"cell_prev",0);
  if (var==NULL) var=dfsVarThisOneSpecial(token,"cell_next",1);
  if (var==NULL){
    nMiddleVariableFree(var);
    dbg("\n\t\t\t\t[dfsVarThisOne] var==NULL, returning");
    return;
  }
  dfsAddToUsedVariables(job,token,var,0,rw,left_of_assignment_expression);
  nMiddleVariableFree(var);
}


// ****************************************************************************
// * dfsVariables
// * On scan pour lister les variables en in/inout/out
// * Par défault, left_of_assignment_expression arrive à 'false'
// ****************************************************************************
void dfsVariables(nablaMain *nabla, nablaJob *job, node *n,
                  bool left_of_assignment_expression){ 
  // Si on hit un assignment et un fils en unary_expression
  // c'est qu'on passe à gauche du '=' et qu'on 'WRITE'
  if (n->ruleid==ruleToId(rule_assignment_expression)&&
      (n->children->ruleid==ruleToId(rule_unary_expression)))
    left_of_assignment_expression=true;
  
  // Si on passe par l'opérateur, on retombe du coté 'READ'
  if (n->ruleid==ruleToId(rule_assignment_operator))
    left_of_assignment_expression=false;

  // Gestion des nabla_system
  if (n->ruleid==ruleToId(rule_nabla_system)){
    // Si on tombe sur la variable systeme TIME,
    // il faut la rajoute au arguments
    if (n->children->tokenid == TIME){
      dbg("\n\t\t[dfsVariables] nabla_system (TIME)!");
      const char *rw=(left_of_assignment_expression)?"WRITE":"READ";
      nablaVariable *var=nMiddleVariableFind(nabla->variables,"time");
      if (var!=NULL)
        dfsAddToUsedVariables(job,"time",var,0,rw,left_of_assignment_expression);
    }
    // Idem pour ITERATION
    if (n->children->tokenid == ITERATION){
      dbg("\n\t\t[dfsVariables] nabla_system (TIME)!");
      const char *rw=(left_of_assignment_expression==true)?"WRITE":"READ";
      nablaVariable *var=nMiddleVariableFind(nabla->variables,"iteration");
      if (var!=NULL)
        dfsAddToUsedVariables(job,"iteration",var,0,rw,left_of_assignment_expression);
    }
    //dfsVarThisOne(nabla,job,n,PREVCELL,"cell_prev",left_of_assignment_expression);
    dfsVarThisOne(nabla,job,n,PREVCELL_X,"cell_prev",left_of_assignment_expression);
    dfsVarThisOne(nabla,job,n,PREVCELL_Y,"cell_prev",left_of_assignment_expression);
    dfsVarThisOne(nabla,job,n,PREVCELL_Z,"cell_prev",left_of_assignment_expression);

    //dfsVarThisOne(nabla,job,n,NEXTCELL,"cell_next",left_of_assignment_expression);
    dfsVarThisOne(nabla,job,n,NEXTCELL_X,"cell_next",left_of_assignment_expression);
    dfsVarThisOne(nabla,job,n,NEXTCELL_Y,"cell_next",left_of_assignment_expression);
    dfsVarThisOne(nabla,job,n,NEXTCELL_Z,"cell_next",left_of_assignment_expression);    
  }

  // Gestion des primary_expression+IDENTIFIER
  if (n->ruleid==ruleToId(rule_primary_expression) &&
      n->children->tokenid == IDENTIFIER){
    dbg("\n\t\t[dfsVariables] primary_expression (%s)!", n->children->token);
    // On cherche un K_OFFSET
    const bool id_has_a_koffset = (n->children->next &&
                                   n->children->next->tokenid==K_OFFSET);

    const int koffset=id_has_a_koffset?getKoffset(n->children->next->token):0;
    if (id_has_a_koffset)
      dbg("\n\t\t[dfsVariables] hum, here what I found on my left: %s (id #%d) offset=%d!",
          n->children->next->token, n->children->next->tokenid, koffset);
    const char *rw=(left_of_assignment_expression)?"WRITE":"READ";
    const char* token = n->children->token;
    // Est-ce une variable connue?
    nablaVariable *var=nMiddleVariableFind(nabla->variables,token);
    if (var){ // Est-ce une variable ?
      dbg("\n\t\t[dfsVariables] Adding!");
      dfsAddToUsedVariables(job,token,var,koffset,rw,left_of_assignment_expression);
      return;
    }else dbg("\n\t\t[dfsVariables] Allready known!");
      
    // Gestion des options
    const nablaOption *opt=nMiddleOptionFindName(nabla->options,token);
    if (opt){
      nablaOption *new=nMiddleOptionFindName(job->used_options,token);
      if (new==NULL){
        // On autorise la modification des options
        //assert(left_of_assignment_expression==false);
        dbg("\n\t\t[dfsVariables] Option '%s' is used (%s) in this job!",opt->name,rw);
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
      return;
    }
  }

  if (n->children)
    dfsVariables(nabla, job, n->children,
                 left_of_assignment_expression);
  if (n->next)
    dfsVariables(nabla, job, n->next,
                 left_of_assignment_expression);
}


// ****************************************************************************
// * dfsEnumMax
// ****************************************************************************
void dfsEnumMax(nablaMain *nabla, nablaJob *job, node *n){
  if (n==NULL) return;
  //if (n->tokenid==FORALL) printf("[1;33m[dfsEnumMax] FORALL[m");
  if (n->tokenid==FORALL && n->next->children->ruleid==ruleToId(rule_forall_switch)){
    if (n->next->children->children->tokenid==CELL)    { job->enum_enum='c'; /*printf("[1;33m cell[m\n");*/}
    if (n->next->children->children->tokenid==CELLS)   { job->enum_enum='c'; /*printf("[1;33m cell[m\n");*/}
    if (n->next->children->children->tokenid==NODE)    { job->enum_enum='n'; /*printf("[1;33m node[m\n");*/}
    if (n->next->children->children->tokenid==NODES)   { job->enum_enum='n'; /*printf("[1;33m node[m\n");*/}
    if (n->next->children->children->tokenid==FACE)    { job->enum_enum='f'; /*printf("[1;33m face[m\n");*/}
    if (n->next->children->children->tokenid==FACES)   { job->enum_enum='f'; /*printf("[1;33m face[m\n");*/}
    if (n->next->children->children->tokenid==PARTICLE){ job->enum_enum='p'; /*printf("[1;33m particle[m\n");*/}
    //exit(printf("[1;33m[dfsEnumMax] job %s, enum_enum=%c[m\n",job->name,job->enum_enum));
  }
  if (n->children!=NULL) dfsEnumMax(nabla,job,n->children);
  if (n->next!=NULL) dfsEnumMax(nabla,job,n->next);
}


// ****************************************************************************
// * dfsExit
// ****************************************************************************
void dfsExit(nablaMain *nabla, nablaJob *job, node *n){
  if (n==NULL) return;  
  if (n->ruleid==ruleToId(rule_nabla_system)
      && (n->children->tokenid==EXIT)){
    job->exists=true;
    //printf("[1;33m[dfsExit] job %s exits[m\n",job->name);
  }
  if (n->children!=NULL) dfsExit(nabla,job,n->children);
  if (n->next!=NULL) dfsExit(nabla,job,n->next);
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
void dfsVariablesDump(nablaMain *nabla, nablaJob *job, node *n){
  nablaVariable *var=job->used_variables;
  dbg("\n\t[dfsVariablesDump]:");
  for(;var!=NULL;var=var->next){
    dbg("\n\t\t[dfsVariablesDump] Variable '%s:(%d)' is used (%s) in this job!",
        var->name,var->koffset,inout(var));
  }
}
