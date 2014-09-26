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

astNode *dfsFetchToken(astNode *n, int tokenid);
astNode *dfsFetchRule(astNode *n, int ruleid);
int dfsScanJobsCalls(void *vars, void *nabla, astNode * n);


int yyUndefTok(void);
int yyTranslate(int tokenid);
int tokenToId(const char *token);
int tokenidToRuleid(int tokenid);
int yyNameTranslate(int tokenid);
int rulenameToId(const char *rulename);

//#define SET_TOKEN(node,newToken) node->token=strdup(newToken)

#define remainY1(lhs)                                                 \
  astNode *timeRemainNode=astNewNodeToken();                          \
  timeRemainNode->token=strdup("slurmTremain()");                     \
  timeRemainNode->tokenid=YYTRANSLATE(REMAIN);                        \
  Y1(lhs,timeRemainNode)

#define limitY1(lhs)                                                  \
  astNode *timeLimitNode=astNewNodeToken();                           \
  timeLimitNode->token=strdup("slurmTlimit()");                       \
  timeLimitNode->tokenid=YYTRANSLATE(LIMIT);                          \
  Y1(lhs,timeLimitNode)

#define volatilePreciseY1(lhs, gmpType){                              \
  astNode *mpTypeNode=astNewNodeToken();                              \
  astNode *volatileNode=astNewNodeToken();                            \
  if (gmpType==GMP_INTEGER)                                           \
    mpTypeNode->token=strdup("mpInteger");                            \
  else                                                                \
    mpTypeNode->token=strdup("mpReal");                               \
  volatileNode->tokenid=VOLATILE;                                     \
  volatileNode->token=strdup("VOLATILE");                             \
  mpTypeNode->tokenid=YYTRANSLATE(gmpType);                           \
  Y2(lhs,mpTypeNode,volatileNode);}

#define preciseY1(lhs, gmpType){                                      \
  astNode *mpTypeNode=astNewNodeToken();                              \
  if (gmpType==GMP_INTEGER)                                           \
    mpTypeNode->token=strdup("mpInteger");                            \
  else                                                                \
    mpTypeNode->token=strdup("mpReal");                               \
  mpTypeNode->tokenid=YYTRANSLATE(gmpType);                           \
  Y1(lhs,mpTypeNode);}

#define primeY1ident(lhs, ident)                                      \
  char token[1024];                                                   \
  astNode *mathlinkPrimeNode=astNewNodeToken();                       \
  sprintf(token, "m_mathlink->Prime(%s);\n\t",ident->token);          \
  mathlinkPrimeNode->token=strdup(token);                             \
  mathlinkPrimeNode->tokenid=YYTRANSLATE(MATHLINK);                   \
  Y1(lhs,mathlinkPrimeNode)

#define primeY1(lhs, cst)                                             \
  char token[1024];                                                   \
  astNode *mathlinkPrimeNode=astNewNodeToken();                       \
  sprintf(token, "m_mathlink->Prime(%s);\n\t",cst->token);            \
  mathlinkPrimeNode->token=strdup(token);                             \
  mathlinkPrimeNode->tokenid=YYTRANSLATE(MATHLINK);                   \
  Y1(lhs,mathlinkPrimeNode)

#define Yp1p(lhs, n1)                                                 \
  astNode *pIn=astNewNodeToken();                                     \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(');               \
  astNode *pOut=astNewNodeToken();                                    \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');             \
  Y3(lhs,pIn,n1,pOut)

#define Yp2p(lhs, n1, n2)                                             \
  astNode *pIn=astNewNodeToken();                                     \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(');               \
  astNode *pOut=astNewNodeToken();                                    \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');             \
  Y4(lhs,n1,pIn,n2,pOut)

#define Ypow(lhs,n1,pow)                                                \
  astNode *pPow=astNewNodeToken();                                       \
  pPow->token=strdup("pow");                                             \
  pPow->tokenid=IDENTIFIER;                                              \
  astNode *pIn=astNewNodeToken();                                       \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(');                 \
  astNode *pTwo=astNewNodeToken();                                      \
  pTwo->token=strdup("," #pow ".0");pTwo->tokenid=IDENTIFIER;                     \
  astNode *pOut=astNewNodeToken();                                      \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');               \
  Y5(lhs,pPow,pIn,n1,pTwo,pOut)

#define Yp3p(lhs, n1, n2, n3)                                         \
  astNode *pIn=astNewNodeToken();                                     \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;              \
  astNode *pOut=astNewNodeToken();                                    \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');             \
  Y5(lhs,pIn,n1,n2,n3,pOut)

// ****************************************************************************
// * FOREACH
// ****************************************************************************
#define Y3_foreach(lhs, n1, n2, n3)                                   \
  astNode *pForeachIni=astNewNodeToken();                             \
  astNode *pForeachEnd=astNewNodeToken();                             \
  pForeachIni->token=strdup("FOREACH_INI");                           \
  pForeachIni->tokenid=FOREACH_INI;                                   \
  pForeachEnd->token=strdup("FOREACH_END");                           \
  pForeachEnd->tokenid=FOREACH_END;                                   \
  Y5(lhs,n1,n2,pForeachIni,n3,pForeachEnd)

#define Y5_foreach(lhs, n1, n2, n3, n4, n5)                           \
  astNode *pForeachIni=astNewNodeToken();                             \
  astNode *pForeachEnd=astNewNodeToken();                             \
  pForeachIni->token=strdup("FOREACH_INI");                           \
  pForeachIni->tokenid=FOREACH_INI;                                   \
  pForeachEnd->token=strdup("FOREACH_END");                           \
  pForeachEnd->tokenid=FOREACH_END;                                   \
  Y7(lhs,n1,n2,pForeachIni,n3,n4,n5,pForeachEnd)

#define Y4_foreach_cell_cell(lhs, n1, n2, n3, n4)                     \
  astNode *pForeachIni=astNewNodeToken();                             \
  astNode *pForeachEnd=astNewNodeToken();                             \
  pForeachIni->token=strdup("FOREACH_INI");                           \
  pForeachIni->tokenid=FOREACH_INI;                                   \
  pForeachEnd->token=strdup("FOREACH_END");                           \
  pForeachEnd->tokenid=FOREACH_END;                                   \
  Y6(lhs,n1,n2,n3,pForeachIni,n4,pForeachEnd)

#define Y4_foreach_cell_node(lhs, n1, n2, n3, n4)                     \
  astNode *pForeachIni=astNewNodeToken();                             \
  astNode *pForeachEnd=astNewNodeToken();                             \
  pForeachIni->token=strdup("FOREACH_INI");                           \
  pForeachIni->tokenid=FOREACH_INI;                                   \
  pForeachEnd->token=strdup("FOREACH_END");                           \
  pForeachEnd->tokenid=FOREACH_END;                                   \
  Y6(lhs,n1,n2,n3,pForeachIni,n4,pForeachEnd)

#define Y4_foreach_cell_face(lhs, n1, n2, n3, n4)                     \
  astNode *pForeachIni=astNewNodeToken();                             \
  astNode *pForeachEnd=astNewNodeToken();                             \
  pForeachIni->token=strdup("FOREACH_INI");                           \
  pForeachIni->tokenid=FOREACH_INI;                                   \
  pForeachEnd->token=strdup("FOREACH_END");                           \
  pForeachEnd->tokenid=FOREACH_END;                                   \
  Y6(lhs,n1,n2,n3,pForeachIni,n4,pForeachEnd)

#define Y4_foreach_cell_particle(lhs, n1, n2, n3, n4)                 \
  astNode *pForeachIni=astNewNodeToken();                             \
  astNode *pForeachEnd=astNewNodeToken();                             \
  pForeachIni->token=strdup("FOREACH_INI");                           \
  pForeachIni->tokenid=FOREACH_INI;                                   \
  pForeachEnd->token=strdup("FOREACH_END");                           \
  pForeachEnd->tokenid=FOREACH_END;                                   \
  Y6(lhs,n1,n2,n3,pForeachIni,n4,pForeachEnd)

#define Yp4p(lhs, n1, n2, n3, n4)                                     \
  astNode *pIn=astNewNodeToken();                                     \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;              \
  astNode *pOut=astNewNodeToken();                                    \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');             \
  Y6(lhs,pIn,n1,n2,n3,n4,pOut)

// Operation '(' Y3 ')'
#define Yop3p(lhs, n1, op, n3)                                          \
  astNode *nOp=astNewNodeToken();                                       \
  nOp->token=strdup(op2name(op->token));                                \
  nOp->tokenid=op->tokenid;                                             \
  astNode *pIn=astNewNodeToken();                                       \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;                \
  astNode *nComa=astNewNodeToken();                                     \
  nComa->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;            \
  astNode *pOut=astNewNodeToken();                                      \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');               \
  Y6(lhs,nOp,pIn,n1,nComa,n3,pOut)

#define Y1(lhs, n1)                                                   \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));

#define Y2(lhs,n1,n2)                                                 \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));

#define Y3(lhs,n1,n2,n3)                                              \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));

#define Y4(lhs,n1,n2,n3,n4)                                           \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));

#define Y5(lhs,n1,n2,n3,n4,n5)                                        \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));

#define YopTernary5p(lhs, cond, question, ifState,doubleDot,elseState)  \
  astNode *nOp=astNewNodeToken();                                       \
  nOp->token=strdup("opTernary");                                       \
  nOp->tokenid=0;                                                       \
  astNode *pIn=astNewNodeToken();                                       \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;                \
  astNode *nComa=astNewNodeToken();                                     \
  nComa->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;            \
  astNode *nComa2=astNewNodeToken();                                    \
  nComa2->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;           \
  astNode *pOut=astNewNodeToken();                                      \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');               \
  Y8(lhs,nOp,pIn,cond,nComa,ifState,nComa2,elseState,pOut)

#define YopDuaryExpression(lhs,ident,op,cond,ifState)                   \
  astNode *nOp=astNewNodeToken();                                       \
  nOp->token=strdup("opTernary");                                       \
  nOp->tokenid=0;                                                       \
  astNode *pIn=astNewNodeToken();                                       \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;                \
  astNode *nComa=astNewNodeToken();                                     \
  nComa->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;            \
  astNode *nComa2=astNewNodeToken();                                    \
  nComa2->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;           \
  astNode *pOut=astNewNodeToken();                                      \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');               \
  astNode *elseState=astNewNodeRule(ident->rule,ident->ruleid);         \
  elseState->children=ident->children;                                  \
  Y10(lhs,ident,op,nOp,pIn,cond,nComa,ifState,nComa2,elseState,pOut)
  


#define Y5_compound_job(lhs,n1,n2,n3,n4,n5)                           \
  astNode *pCompoundJobEnd=astNewNodeToken();                         \
  astNode *pCompoundJobIni=astNewNodeToken();                         \
  pCompoundJobIni->token=strdup("COMPOUND_JOB_INI");                  \
  pCompoundJobIni->tokenid=COMPOUND_JOB_INI;                          \
  pCompoundJobEnd->token=strdup("nabla_job_definition_end");          \
  pCompoundJobEnd->tokenid=COMPOUND_JOB_END;                          \
  Y7(lhs,n1,n2,n3,n4,pCompoundJobIni,n5,pCompoundJobEnd)

#define Y6(lhs,n1,n2,n3,n4,n5,n6)                                     \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));                                              \
  astAddNext((n5),(n6));
#define Y6_compound_job(lhs,n1,n2,n3,n4,n5,n6)                        \
  astNode *pCompoundJobEnd=astNewNodeToken();                         \
  astNode *pCompoundJobIni=astNewNodeToken();                         \
  pCompoundJobIni->token=strdup("COMPOUND_JOB_INI");                  \
  pCompoundJobIni->tokenid=COMPOUND_JOB_INI;                          \
  pCompoundJobEnd->token=strdup("nabla_job_definition_end");          \
  pCompoundJobEnd->tokenid=COMPOUND_JOB_END;                          \
  Y8(lhs,n1,n2,n3,n4,n5,pCompoundJobIni,n6,pCompoundJobEnd)

#define Y7(lhs,n1,n2,n3,n4,n5,n6,n7)                                  \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));                                              \
  astAddNext((n5),(n6));                                              \
  astAddNext((n6),(n7));

#define Y7_compound_job(lhs, n1, n2, n3, n4, n5,n6,n7)                \
  astNode *pCompoundJobEnd=astNewNodeToken();                         \
  astNode *pCompoundJobIni=astNewNodeToken();                         \
  pCompoundJobIni->token=strdup("COMPOUND_JOB_INI");                  \
  pCompoundJobIni->tokenid=COMPOUND_JOB_INI;                          \
  pCompoundJobEnd->token=strdup("COMPOUND_JOB_END");          \
  pCompoundJobEnd->tokenid=COMPOUND_JOB_END;                          \
  Y9(lhs,n1,n2,n3,n4,n5,n6,pCompoundJobIni,n7,pCompoundJobEnd)

#define Y8(lhs,n1,n2,n3,n4,n5,n6,n7,n8)                               \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));                                              \
  astAddNext((n5),(n6));                                              \
  astAddNext((n6),(n7));                                              \
  astAddNext((n7),(n8));
#define Y8_compound_job(lhs,n1,n2,n3,n4,n5,n6,n7,n8)                  \
  astNode *pCompoundJobEnd=astNewNodeToken();                         \
  astNode *pCompoundJobIni=astNewNodeToken();                         \
  pCompoundJobIni->token=strdup("COMPOUND_JOB_INI");                  \
  pCompoundJobIni->tokenid=COMPOUND_JOB_INI;                          \
  pCompoundJobEnd->token=strdup("COMPOUND_JOB_END");                  \
  pCompoundJobEnd->tokenid=COMPOUND_JOB_END;                          \
  Y10(lhs,n1,n2,n3,n4,n5,n6,n7,pCompoundJobIni,n8,pCompoundJobEnd)

#define Y9(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9)                            \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));                                              \
  astAddNext((n5),(n6));                                              \
  astAddNext((n6),(n7));                                              \
  astAddNext((n7),(n8));                                              \
  astAddNext((n8),(n9));

#define Y10(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10)                       \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));                                              \
  astAddNext((n5),(n6));                                              \
  astAddNext((n6),(n7));                                              \
  astAddNext((n7),(n8));                                              \
  astAddNext((n8),(n9));                                              \
  astAddNext((n9),(n10));


#define Y11(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11)                   \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));                                              \
  astAddNext((n5),(n6));                                              \
  astAddNext((n6),(n7));                                              \
  astAddNext((n7),(n8));                                              \
  astAddNext((n8),(n9));                                              \
  astAddNext((n9),(n10));                                             \
  astAddNext((n10),(n11));
#define Y11_compound_job(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11)        \
  astNode *pCompoundJobEnd=astNewNodeToken();                           \
  astNode *pCompoundJobIni=astNewNodeToken();                           \
  pCompoundJobIni->token=strdup("COMPOUND_JOB_INI");                    \
  pCompoundJobIni->tokenid=COMPOUND_JOB_INI;                            \
  pCompoundJobEnd->token=strdup("COMPOUND_JOB_END");                    \
  pCompoundJobEnd->tokenid=COMPOUND_JOB_END;                            \
  Y13(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,pCompoundJobIni,n11,pCompoundJobEnd)

#define Y12(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)               \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));                                              \
  astAddNext((n5),(n6));                                              \
  astAddNext((n6),(n7));                                              \
  astAddNext((n7),(n8));                                              \
  astAddNext((n8),(n9));                                              \
  astAddNext((n9),(n10));                                             \
  astAddNext((n10),(n11));                                            \
  astAddNext((n11),(n12));
#define Y12_compound_job(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)    \
  astNode *pCompoundJobEnd=astNewNodeToken();                           \
  astNode *pCompoundJobIni=astNewNodeToken();                           \
  pCompoundJobIni->token=strdup("COMPOUND_JOB_INI");                    \
  pCompoundJobIni->tokenid=COMPOUND_JOB_INI;                            \
  pCompoundJobEnd->token=strdup("COMPOUND_JOB_END");                    \
  pCompoundJobEnd->tokenid=COMPOUND_JOB_END;                            \
  Y14(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,pCompoundJobIni,n12,pCompoundJobEnd)


#define Y13(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13)           \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));                                              \
  astAddNext((n5),(n6));                                              \
  astAddNext((n6),(n7));                                              \
  astAddNext((n7),(n8));                                              \
  astAddNext((n8),(n9));                                              \
  astAddNext((n9),(n10));                                             \
  astAddNext((n10),(n11));                                            \
  astAddNext((n11),(n12));                                            \
  astAddNext((n12),(n13));


#define Y14(lhs,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14)       \
  lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);                   \
  astAddChild((lhs),(n1));                                            \
  astAddNext((n1),(n2));                                              \
  astAddNext((n2),(n3));                                              \
  astAddNext((n3),(n4));                                              \
  astAddNext((n4),(n5));                                              \
  astAddNext((n5),(n6));                                              \
  astAddNext((n6),(n7));                                              \
  astAddNext((n7),(n8));                                              \
  astAddNext((n8),(n9));                                              \
  astAddNext((n9),(n10));                                             \
  astAddNext((n10),(n11));                                            \
  astAddNext((n11),(n12));                                            \
  astAddNext((n12),(n13));                                            \
  astAddNext((n13),(n14));


#endif // _NABLA_AST_H_
