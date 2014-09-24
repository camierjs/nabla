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

/*
 * Creat a new TOKEN node, initialize its contents
 */
astNode *astNewNodeToken(void) {
	astNode *n;
	n = (astNode *)malloc(sizeof(astNode));
	assert(n != NULL);
   // token & tokenid are set by the LEXER: //static int tok(YYSTYPE *yylval, int tokenid)
   n->rule=NULL;
   n->ruleid=0;
   n->next=n->children=n->parent=NULL;
  	return n; 
}


/* 
 * Creat a new RULE node, initialize its contents
 */
astNode * astNewNodeRule(const char *rule, unsigned int yyn) {
	astNode * n;
	//dbg("\n\t[astNewNodeRule] %s %d", rule, yyn);
	n = (astNode *)malloc(sizeof(astNode));
	assert(n != NULL);
   n->token=NULL;
   n->tokenid=0;
   n->next=n->children=n->parent=NULL;
	assert(rule != NULL);
	n->rule = rule; 
	n->ruleid = yyn;   
	return n;
}

/* astAddChild:
 *
 * Add a child to a parent node. If the child is linked to something else, keep the link.
 * Append the child as the last of the parents children.
 * Returns a pointer to the parent. Parameters are pointers to the parent node and the
 * child node to add.
 */
astNode *astAddChild(astNode *parent, astNode *child) {
	astNode *iterator = NULL;
	assert(parent != NULL && child != NULL);
   child->parent=parent;
	iterator = parent->children;
	if(iterator == NULL)
		parent->children = child;
   else{
		while(iterator->next != NULL)
			iterator = iterator->next;
		iterator->next = child;
	}
	return parent;
}


/* astAddNext:
 *
 * Add a sibling-to-be node to a node, appending it as the last sibling (recursively).
 * Parameters are pointers to the node and to the sibling-to-be. Return value is a pointer
 * to the node. The return value does not need to be checked.
 */
astNode *astAddNext(astNode *node, astNode *next) {
	assert(node != NULL && next != NULL);
	if(node->next != NULL)
		astAddNext(node->next, next);
	else{
		node->next = next;
      next->parent=node->parent;
   }
	return node;
}

