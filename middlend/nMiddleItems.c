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
 * type_specifier
 *****************************************************************************/
static void actItemTypeSpecifier(astNode * n, void *generic_arg){
  nablaMain *arc=(nablaMain*)generic_arg;
  nablaVariable *variable = nMiddleVariableNew(arc);
  dbg("\n\t\t[actItemTypeSpecifier] %s:%s", arc->tmpVarKinds, n->children->token);
  // On regarde s'il n'y a pas un noeud à coté qui nous dit de ne pas backuper
  if (n->children->next != NULL &&  n->children->next->tokenid==VOLATILE){
    dbg("\n\t\t[actItemTypeSpecifier] %s, id=%d volatile(%d) hit!",
        n->children->next->token, n->children->next->tokenid, VOLATILE);
    variable->dump=false;
  }else{
    variable->dump=true;
  }
  nMiddleVariableAdd(arc, variable);
  variable->item=strdup(arc->tmpVarKinds);
  variable->type=toolStrDownCase(n->children->token);
  // Par défaut, on met à '0' la dimension de la variable
  variable->dim=0;
  // Si on a un gmp precise integer, on dit que c'est un tableau de 'byte'
  if (strcmp(variable->type, "mpinteger")==0){
    variable->gmpRank=nMiddleVariableGmpRank(arc->variables);
    dbg("\n\t\t[actItemTypeSpecifier] Found GMP rank=%d", variable->gmpRank);
    variable->type=strdup("integer");
    variable->dim=1;
  }
 }


/***************************************************************************** 
 * direct_declarator
 *****************************************************************************/
static void actItemDirectDeclarator(astNode * n, void *generic_arg){
  nablaMain *arc=(nablaMain*)generic_arg;
  nablaVariable *variable =nMiddleVariableLast(arc->variables);
  dbg("\n\t\t[actItemDirectDeclarator] %s", n->children->token);
  variable->name=strdup(n->children->token);
  if (variable->gmpRank!=-1)
    dbg("\n\t\t[actItemTypeSpecifier] Found GMP variable %s", variable->name);
}


// ***************************************************************************** 
// * actItemNablaItems
// *****************************************************************************
static void actItemNablaItems(astNode * n, void *generic_arg){
  nablaMain *arc=(nablaMain*)generic_arg;
  nablaVariable *variable =nMiddleVariableLast(arc->variables);
  const char cfgn=variable->item[0];
  dbg("\n\t\t[actItemNablaItems] variable '%s' on '%s' array of [%s]",
      variable->name, variable->item, n->children->token);
  // Si on tombe sur un nabla_item ici, c'est que c'est un tableau à la dimension de cet item
  variable->dim=1;
  variable->size=0;
  if (cfgn=='n'&& n->children->tokenid==CELLS) variable->size=8; // node_var[cells]: tied @ 8 (hex mesh)
  if (cfgn=='f'&& n->children->tokenid==CELLS) variable->size=2; // face_var[cells]: backCell & frontCell
//#warning cell_var[nodes]: tied @ 8 (hex mesh), should depend on domension
  if (cfgn=='c'&& n->children->tokenid==NODES) variable->size=8; // cell_var[nodes]: tied @ 8 (hex mesh), should depend on dimension
  if (n->children->tokenid==FACES) variable->size=8;
  if (n->children->tokenid==PARTICLES) variable->size=8;
  if (n->children->tokenid==MATERIALS) variable->size=8;
  if (n->children->tokenid==ENVIRONMENTS) variable->size=8;
  dbg("\n\t\t[actItemNablaItems] variable->size=%d", variable->size);
}


//***************************************************************************** 
// * actItemPrimaryExpression
// ****************************************************************************
static void actItemPrimaryExpression(astNode * n, void *generic_arg){
  nablaMain *arc=(nablaMain*)generic_arg;
  nablaVariable *variable =nMiddleVariableLast(arc->variables);
  dbg("\n\t\t[actItemPrimaryExpression] %s", n->children->token);
  // Si on tombe sur un primary_expression ici, c'est que c'est un tableau à la dimension constante
  variable->dim=1;
  variable->size=atol(n->children->token);
  dbg("\n\t\t[actItemPrimaryExpression] variable->size=%d", variable->size);
}


//***************************************************************************** 
// * actItemList
// ****************************************************************************
__attribute__((unused)) static void actItemList(astNode * n, void *generic_arg){
  dbg("\n\t\t[actItemList]");
}


//***************************************************************************** 
// * actItemComa
// ****************************************************************************
static void actItemComa(astNode * n, void *generic_arg){
  dbg("\n\t\t[actItemComa]");
  nablaMain *arc=(nablaMain*)generic_arg;
  nablaVariable *variable = nMiddleVariableNew(arc);
  // On va chercher la dernière variable créée
  nablaVariable *last_variable = nMiddleVariableLast(arc->variables);
  variable->dump=true;
  nMiddleVariableAdd(arc, variable);
  variable->item=strdup(arc->tmpVarKinds);
  // On utilise le type de la dernière variable pour cette nouvelle
  variable->type=last_variable->type;
  variable->dim=0;
}


/***************************************************************************** 
 * Scan pour la déclaration des variables
 *****************************************************************************/
void nMiddleItems(astNode * n, int ruleid, nablaMain *arc){
  RuleAction tokact[]={
    {rulenameToId("type_specifier"),actItemTypeSpecifier},
    //{rulenameToId("nabla_direct_declarator_list") ,actItemList},
    {tokenidToRuleid(',') ,actItemComa},
    {rulenameToId("nabla_direct_declarator"),actItemDirectDeclarator},
    {rulenameToId("nabla_items"), actItemNablaItems},
    {rulenameToId("primary_expression"), actItemPrimaryExpression},
    {0,NULL}
  };
  if (n==NULL) return;
  if(n->rule != NULL)
    if (ruleid ==  n->ruleid)
      scanTokensForActions(n, tokact, (void*)arc);
  nMiddleItems(n->children, ruleid, arc);
  nMiddleItems(n->next, ruleid, arc);
}

