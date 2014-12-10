// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
#include "nabla.h"

nablaOption *nablaOptionNew(nablaMain *nabla){
	nablaOption *option;
	option = (nablaOption *)malloc(sizeof(nablaOption));
 	assert(option != NULL);
   option->axl_it=true; // Par défaut, on dump la option dans le fichier AXL
   option->type=option->name=option->dflt=NULL;
   option->main=nabla;
   option->next=NULL;
  	return option; 
}

nablaOption *nablaOptionLast(nablaOption *options) {
   while(options->next != NULL){
     options = options->next;
   }
   return options;
}

nablaOption *nablaOptionAdd(nablaMain *nabla, nablaOption *option) {
  assert(option != NULL);
  if (nabla->options == NULL)
    nabla->options=option;
  else
    nablaOptionLast(nabla->options)->next=option;
  return NABLA_OK;
}


/*
 * 
 */
nablaOption *findOptionName(nablaOption *options, char *name) {
  nablaOption *option=options;
  //dbg("\n\t[findOptionName] %s", name);
  //assert(option != NULL && name != NULL);
  if (option==NULL) return NULL;
  while(option != NULL) {
    //dbg(" ?%s", option->name);
    if(strcmp(option->name, name) == 0){
      //dbg(" Yes!");
      return option;
    }
    option = option->next;
  }
  //dbg(" Nope!");
  return NULL;
}


/***************************************************************************** 
 * type_specifier
 *****************************************************************************/
static void actOptionsTypeSpecifier(astNode * n, void *generic_arg){
  nablaMain *nabla=(nablaMain*)generic_arg;
  nablaOption *option = nablaOptionNew(nabla);
  dbg("\n\t\t[actGenericOptionsTypeSpecifier] %s",n->children->token);
  nablaOptionAdd(nabla, option);
  option->type=toolStrDownCase(n->children->token);
}


/***************************************************************************** 
 * direct_declarator
 *****************************************************************************/
static void actOptionsDirectDeclarator(astNode * n, void *generic_arg){
  nablaMain *nabla=(nablaMain*)generic_arg;
  nablaOption *option =nablaOptionLast(nabla->options);
  dbg("\n\t\t[actGenericOptionsDirectDeclarator] %s", n->children->token);
  option->name=strdup(n->children->token);
}


/***************************************************************************** 
 * primary_expression
 *****************************************************************************/
void catTillToken(astNode * n, char *dflt){
  if (n==NULL) return;
  if (n->token != NULL){
    dbg("\n\t\t\t[catTillToken] %s", n->token);
    if (n->tokenid != ';'){
      dflt=realloc(dflt, strlen(dflt)+strlen(n->token));
      strcat(dflt,n->token);
    }
  }
  if (n->children != NULL) catTillToken(n->children, dflt);
  if (n->next != NULL) catTillToken(n->next, dflt);
}


/*static void actOptionsPrimaryExpression(astNode * n, void *generic_arg){
  nablaMain *nabla=(nablaMain*)generic_arg;
  nablaOption *option =nablaOptionLast(nabla->options);
  dbg("\n\t\t[actOptionsPrimaryExpression] %s", n->children->token);
  option->dflt=strdup(n->children->token);
  catTillToken(n->children->next, option->dflt);
  dbg("\n\t\t[actOptionsPrimaryExpression] final option->dflt is '%s'", option->dflt);
  }*/

static void actOptionsExpression(astNode * n, void *generic_arg){
  nablaMain *nabla=(nablaMain*)generic_arg;
  nablaOption *option =nablaOptionLast(nabla->options);
  dbg("\n\t\t[actOptionsExpression] rule=%s", n->rule);
  option->dflt=strdup("");
  catTillToken(n->children, option->dflt);
  dbg("\n\t\t[actOptionsExpression] final option->dflt is '%s'", option->dflt);
}




/***************************************************************************** 
 * Scan pour la déclaration des options
 *****************************************************************************/
void nablaOptions(astNode * n, int ruleid, nablaMain *nabla){
  RuleAction tokact[]={
    {rulenameToId("type_specifier"),actOptionsTypeSpecifier},
    {rulenameToId("direct_declarator"),actOptionsDirectDeclarator},
    {rulenameToId("expression"),actOptionsExpression},
    {0,NULL}};
  assert(ruleid!=1);
  if (n->rule != NULL)
    if (ruleid ==  n->ruleid)
      scanTokensForActions(n, tokact, (void*)nabla);
  if (n->children != NULL) nablaOptions(n->children, ruleid, nabla);
  if (n->next != NULL) nablaOptions(n->next, ruleid, nabla);
}


/*****************************************************************************
 * Transformation de tokens en options
 *****************************************************************************/
nablaOption *turnTokenToOption(astNode * n, nablaMain *arc){
  nablaOption *opt=findOptionName(arc->options, n->token);
  // Si on ne trouve pas d'option, on a rien à faire
  if (opt == NULL) return NULL;
  arc->hook->turnTokenToOption(arc,opt);
  return opt;
}
