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
#include "nabla.tab.h"


/***************************************************************************** 
 * type_specifier
 *****************************************************************************/
static void actItemTypeSpecifier(astNode * n, void *generic_arg){
  nablaMain *arc=(nablaMain*)generic_arg;
  nablaVariable *variable = nablaVariableNew(arc);
  dbg("\n\t\t[actItemTypeSpecifier] %s:%s", arc->tmpVarKinds, n->children->token);
  // On regarde s'il n'y a pas un noeud à coté qui nous dit de ne pas backuper
  if (n->children->next != NULL &&  n->children->next->tokenid==VOLATILE){
    dbg("\n\t\t[actItemTypeSpecifier] %s, id=%d volatile(%d) hit!",
        n->children->next->token, n->children->next->tokenid, VOLATILE);
    variable->dump=false;
  }else{
    variable->dump=true;
  }
  nablaVariableAdd(arc, variable);
  variable->item=strdup(arc->tmpVarKinds);
  variable->type=toolStrDownCase(n->children->token);
  // Par défaut, on met à '0' la dimension de la variable
  variable->dim=0;
  // Si on a un gmp precise integer, on dit que c'est un tableau de 'byte'
  if (strcmp(variable->type, "mpinteger")==0){
    variable->gmpRank=nablaVariableGmpRank(arc->variables);
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
  nablaVariable *variable =nablaVariableLast(arc->variables);
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
  nablaVariable *variable =nablaVariableLast(arc->variables);
  dbg("\n\t\t[actItemNablaItems] %s", n->children->token);
  // Si on tombe sur un nabla_item ici, c'est que c'est un tableau à la dimension de cet item
  variable->dim=1;
  variable->size=0;
  if (n->children->tokenid==CELLS) variable->size=8;
  if (n->children->tokenid==NODES) variable->size=8;
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
  nablaVariable *variable =nablaVariableLast(arc->variables);
  dbg("\n\t\t[actItemPrimaryExpression] %s", n->children->token);
  // Si on tombe sur un primary_expression ici, c'est que c'est un tableau à la dimension constante
  variable->dim=1;
  variable->size=atol(n->children->token);
  dbg("\n\t\t[actItemPrimaryExpression] variable->size=%d", variable->size);
}


/***************************************************************************** 
 * Scan pour la déclaration des variables
 *****************************************************************************/
void nablaItems(astNode * n, int ruleid, nablaMain *arc){
  RuleAction tokact[]={
    {rulenameToId("type_specifier"),actItemTypeSpecifier},
    {rulenameToId("nabla_direct_declarator"),actItemDirectDeclarator},
    {rulenameToId("nabla_items"), actItemNablaItems},
    {rulenameToId("primary_expression"), actItemPrimaryExpression},
    {0,NULL}
  };
  if (n==NULL) return;
  //assert(ruleid!=1);
  if(n->rule != NULL)
    if (ruleid ==  n->ruleid)
      scanTokensForActions(n, tokact, (void*)arc);
  nablaItems(n->children, ruleid, arc);
  nablaItems(n->next, ruleid, arc);
}

