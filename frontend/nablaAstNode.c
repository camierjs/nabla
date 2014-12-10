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

// ****************************************************************************
// * astNewNode
// ****************************************************************************
astNode *astNewNode(void) {
  astNode *n = (astNode*)calloc(1,sizeof(astNode));
  assert(n != NULL);
  return n; 
}



// ****************************************************************************
// * astNewNodeRule
// ****************************************************************************
astNode *astNewNodeRule(const char *rule, unsigned int yyn) {
  astNode *n=astNewNode();
  assert(rule != NULL);
  n->rule = rule; 
  n->ruleid = yyn;   
  return n;
}


// ****************************************************************************
// * astAddChild
// ****************************************************************************
astNode *astAddChild(astNode *root, astNode *child) {
  assert(root != NULL && child != NULL);
  astNode *next=root->children;
  // On set le parent du nouvel enfant
  child->parent=root;
  // S'il n'y a pas de fils, on le crée
  if (root->children==NULL) return root->children = child;
  // Sinon, on scrute jusqu'au dernier enfant
  for(;next->next!=NULL;next=next->next);
  // Et on l'append
  return next->next=child;
}


// ****************************************************************************
// * astAddNext
// ****************************************************************************
astNode *astAddNext(astNode *root, astNode *node) {
  assert(root != NULL && node != NULL);
  astNode *next=root;
  // On set le parent du nouvel enfant
  node->parent=root->parent;
  // S'il n'y a pas de frère, on le crée
  if(root->next == NULL) return root->next = node;
  // Sinon, on scrute jusqu'au dernier enfant
  for(;next->next!=NULL;next=next->next);
  // Et on l'append
  return next->next=node;
}

