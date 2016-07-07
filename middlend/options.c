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


// ****************************************************************************
// * nMiddleOptionFree
// ****************************************************************************
void nMiddleOptionFree(nablaOption *options){
  for(nablaOption *this,*option=this=options;option!=NULL;free(this))
    option=(this=option)->next;
}

// ****************************************************************************
// * nMiddleOption: New, Last & Add
// ****************************************************************************
nablaOption *nMiddleOptionNew(nablaMain *nabla){
	nablaOption *option;
	option = (nablaOption *)calloc(1,sizeof(nablaOption));
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


// ****************************************************************************
// * nMiddleOptionFindName
// ****************************************************************************
nablaOption *nMiddleOptionFindName(nablaOption *options, const char *name) {
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


// ****************************************************************************
// * nMiddleCatTillToken
// ****************************************************************************
static void nMiddleCatTillToken(astNode * n, char **dflt){
  if (n==NULL) return;
  if (n->token != NULL)
    if (n->tokenid != ';')
      strcat(*dflt,n->token);
  if (n->children != NULL) nMiddleCatTillToken(n->children, dflt);
  if (n->next != NULL) nMiddleCatTillToken(n->next, dflt);
}


// ****************************************************************************
// * type_specifier
// ****************************************************************************
static void actOptionsTypeSpecifier(astNode * n, void *generic_arg){
  nablaMain *nabla=(nablaMain*)generic_arg;
  nablaOption *option = nMiddleOptionNew(nabla);
  dbg("\n\t\t[actGenericOptionsTypeSpecifier] %s",n->children->token);
  nMiddleOptionAdd(nabla, option);
  option->type=toolStrDownCase(n->children->token);
}


// ****************************************************************************
// * direct_declarator
// ****************************************************************************
static void actOptionsDirectDeclarator(astNode * n, void *generic_arg){
  nablaMain *nabla=(nablaMain*)generic_arg;
  nablaOption *option =nMiddleOptionLast(nabla->options);
  dbg("\n\t\t[actGenericOptionsDirectDeclarator] %s", n->children->token);
  option->name=sdup(n->children->token);
  //printf("\noption: %s",n->children->token);
  //option->name=utf2ascii(n->children->token);
}


// ****************************************************************************
// * primary_expression
// ****************************************************************************
static void actOptionsExpression(astNode * n, void *generic_arg){
  nablaMain *nabla=(nablaMain*)generic_arg;
  nablaOption *option =nMiddleOptionLast(nabla->options);
  dbg("\n\t\t[actOptionsExpression] rule=%s", n->rule);
  char *dflt=(char*)calloc(NABLA_MAX_FILE_NAME,sizeof(char));
  nMiddleCatTillToken(n->children, &dflt);
  option->dflt=sdup(dflt);
  free(dflt);
  dbg("\n\t\t[actOptionsExpression] final option->dflt is '%s'", option->dflt);
}


// ****************************************************************************
// * Scan pour la déclaration des options
// ****************************************************************************
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


// ****************************************************************************
// * Transformation de tokens en options
// ****************************************************************************
nablaOption *nMiddleTurnTokenToOption(astNode * n, nablaMain *nabla){
  nablaOption *opt=nMiddleOptionFindName(nabla->options, n->token);
  // Si on ne trouve pas d'option, on a rien à faire
  if (opt == NULL) return NULL;
  nabla->hook->token->option(nabla,opt);
  return opt;
}
