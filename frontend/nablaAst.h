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

astNode *astNewNode(void);
astNode *astNewNodeRule(const char*, unsigned int);
astNode *astAddChild(astNode*, astNode*);
astNode *astAddNext(astNode*, astNode*);

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
