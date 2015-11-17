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
#ifndef _NABLA_AST_H_
#define _NABLA_AST_H_


// ****************************************************************************
// * Node structure used for the AST
// ****************************************************************************
typedef struct astNodeStruct{
  unsigned int id; // Unique node ID
  char *token;
  char *token_utf8;
  int tokenid;
  const char * rule;
  int ruleid;
  bool type_name;
  struct astNodeStruct *next, *children, *parent;  
}astNode;


// ****************************************************************************
// * Structure used to pass a batch of actions for each ruleid found while DFS
// ****************************************************************************
typedef struct RuleActionStruct{
  int ruleid;
  void (*action)(astNode*,void*);
} RuleAction;


// ****************************************************************************
// * Forward declaration of AST functions
// ****************************************************************************
astNode *astNewNode(void);
astNode *astNewNodeRule(const char*, unsigned int);
astNode *astAddChild(astNode*, astNode*);
astNode *astAddNext(astNode*, astNode*);

NABLA_STATUS astTreeSave(const char*, astNode*);
void getInOutPutsNodes(FILE*, astNode *n, char *color);
void getInOutPutsEdges(FILE*, astNode *n, int inout, char *nName1, char* nName2);

void dfsUtf8(astNode*);
void dfsDumpToken(astNode*);

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
