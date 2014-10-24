/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccAstNode.c																	  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 07.01.2010																	  *
 * Updated  : 12.11.2012																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 07.01.2010	jscamier	Creation															  *
 *****************************************************************************/
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

