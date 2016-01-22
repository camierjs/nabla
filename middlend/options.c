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

nablaOption *nMiddleOptionNew(nablaMain *nabla){
	nablaOption *option;
	option = (nablaOption *)malloc(sizeof(nablaOption));
 	assert(option != NULL);
   option->axl_it=true; // Par défaut, on dump la option dans le fichier AXL
   option->type=option->name=option->dflt=NULL;
   option->main=nabla;
   option->next=NULL;
  	return option; 
}

nablaOption *nMiddleOptionLast(nablaOption *options) {
   while(options->next != NULL){
     options = options->next;
   }
   return options;
}

nablaOption *nMiddleOptionAdd(nablaMain *nabla, nablaOption *option) {
  assert(option != NULL);
  if (nabla->options == NULL)
    nabla->options=option;
  else
    nMiddleOptionLast(nabla->options)->next=option;
  return NABLA_OK;
}


/*
 * 
 */
nablaOption *nMiddleOptionFindName(nablaOption *options, char *name) {
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
  nablaOption *option = nMiddleOptionNew(nabla);
  dbg("\n\t\t[actGenericOptionsTypeSpecifier] %s",n->children->token);
  nMiddleOptionAdd(nabla, option);
  option->type=toolStrDownCase(n->children->token);
}


/***************************************************************************** 
 * direct_declarator
 *****************************************************************************/
static void actOptionsDirectDeclarator(astNode * n, void *generic_arg){
  nablaMain *nabla=(nablaMain*)generic_arg;
  nablaOption *option =nMiddleOptionLast(nabla->options);
  dbg("\n\t\t[actGenericOptionsDirectDeclarator] %s", n->children->token);
  option->name=strdup(n->children->token);
}


/***************************************************************************** 
 * primary_expression
 *****************************************************************************/
void nMiddleCatTillToken(astNode * n, char **dflt){
  if (n==NULL) return;
  if (n->token != NULL){
    dbg("\n\t\t\t[catTillToken] %s", n->token);
    if (n->tokenid != ';'){
      *dflt=realloc(*dflt, strlen(*dflt)+strlen(n->token));
      strcat(*dflt,n->token);
    }
  }
  if (n->children != NULL) nMiddleCatTillToken(n->children, dflt);
  if (n->next != NULL) nMiddleCatTillToken(n->next, dflt);
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
  nablaOption *option =nMiddleOptionLast(nabla->options);
  dbg("\n\t\t[actOptionsExpression] rule=%s", n->rule);
  option->dflt=calloc(1024,1);
  nMiddleCatTillToken(n->children, &option->dflt);
  dbg("\n\t\t[actOptionsExpression] final option->dflt is '%s'", option->dflt);
}




/***************************************************************************** 
 * Scan pour la déclaration des options
 *****************************************************************************/
void nMiddleOptions(astNode * n, int ruleid, nablaMain *nabla){
  RuleAction tokact[]={
    {rulenameToId("type_specifier"),actOptionsTypeSpecifier},
    {rulenameToId("direct_declarator"),actOptionsDirectDeclarator},
    {rulenameToId("expression"),actOptionsExpression},
    {0,NULL}};
  assert(ruleid!=1);
  if (n->rule != NULL)
    if (ruleid ==  n->ruleid)
      scanTokensForActions(n, tokact, (void*)nabla);
  if (n->children != NULL) nMiddleOptions(n->children, ruleid, nabla);
  if (n->next != NULL) nMiddleOptions(n->next, ruleid, nabla);
}


/*****************************************************************************
 * Transformation de tokens en options
 *****************************************************************************/
nablaOption *nMiddleTurnTokenToOption(astNode * n, nablaMain *arc){
  nablaOption *opt=nMiddleOptionFindName(arc->options, n->token);
  // Si on ne trouve pas d'option, on a rien à faire
  if (opt == NULL) return NULL;
  arc->hook->token->option(arc,opt);
  return opt;
}
