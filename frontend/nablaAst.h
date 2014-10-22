/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccAst.h	   																  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 07.01.2010																	  *
 * Updated  : 07.01.2010																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 07.01.2010	jscamier	Creation															  *
 *****************************************************************************/
#ifndef _NABLA_AST_H_
#define _NABLA_AST_H_

typedef struct astNodeStruct{
  unsigned int id;
  char *token;
  char *token_utf8;
  int tokenid;
  const char * rule;
  int ruleid;
  struct astNodeStruct *next, *children, *parent;  
}astNode;

typedef struct RuleActionStruct{
  int ruleid;
  void (*action)(astNode*,void*);
} RuleAction;

astNode *astNewNodeToken(void);
astNode *astNewNodeRule(const char *, unsigned int);
astNode *astAddChild(astNode *parent, astNode *child);
astNode *astAddNext(astNode *node, astNode *next);

NABLA_STATUS astTreeSave(const char *treeFileName, astNode *root);
void getInOutPutsNodes(FILE *fOut, astNode *n, char *color);
void getInOutPutsEdges(FILE *fOut, astNode *n, int inout, char *nName1, char* nName2);

void dfsUtf8(astNode*);

void scanTokensForActions(astNode * n, RuleAction *tokact, void *arc);
char *dfsFetchFirst(astNode *n, int ruleid);
astNode *dfsFetch(astNode *n, int ruleid);

astNode *dfsFetchTokenId(astNode *n, int tokenid);
astNode *dfsFetchToken(astNode *n, const char *token);
astNode *dfsFetchRule(astNode *n, int ruleid);
int dfsScanJobsCalls(void *vars, void *nabla, astNode * n);


int yyUndefTok(void);
int yyTranslate(int tokenid);
int tokenToId(const char *token);
int tokenidToRuleid(int tokenid);
int yyNameTranslate(int tokenid);
int rulenameToId(const char *rulename);

#endif // _NABLA_AST_H_
