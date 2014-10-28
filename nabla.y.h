/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nablaY.h	   																  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2014.10.22																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 *****************************************************************************/
#ifndef _NABLA_Y_H_
#define _NABLA_Y_H_


// ****************************************************************************
// * COUNTING VARIADIC ARGUMENTS
// ****************************************************************************
#define __NB_ARGS__(z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,cnt,...) cnt
#define NB_ARGS(...) __NB_ARGS__(,##__VA_ARGS__,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)
#define TRY_TO_COUNT_HERE(...) __NB_ARGS__(__VA_ARGS__)


// ****************************************************************************
// * Forward declarations
// ****************************************************************************
void rhsAdd(astNode**,int,astNode**);
void rhsAddVariadic(astNode**,int,int,...);

void rhsTailSandwich(astNode**,int,int,int,astNode**);
void rhsTailSandwichVariadic(astNode**,int,int,int,int,...);


// ****************************************************************************
// * RHS
// ****************************************************************************
#define rhs rhsAdd(&yyval,yyn,yyvsp);
#define RHS(lhs, ...) rhsAddVariadic(&yyval,yyn,NB_ARGS(__VA_ARGS__),__VA_ARGS__)


// ****************************************************************************
// * Foreach
// ****************************************************************************
#define foreach rhsTailSandwich(&yyval,yyn,FOREACH_INI,FOREACH_END,yyvsp);
#define foreachVariadic(lhs, ...)                                        \
  tailSandwichVariadic(FOREACH_INI,FOREACH_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)


// ****************************************************************************
// * Compound Jobs
// ****************************************************************************
#define compound_job(lhs,...)                                           \
  tailSandwich(COMPOUND_JOB_INI,COMPOUND_JOB_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)

#define compound_job_without__NB_ARGS__(lhs,...)                        \
  assert(NB_ARGS(__VA_ARGS__)==yyr2[yyn]);                              \
  tailSandwich(COMPOUND_JOB_INI,COMPOUND_JOB_END,yyr2[yyn],__VA_ARGS__)


// *****************************************************************************
// * Tail Sandwich
// * printf("[1;33mYp2p=%d[m\n",yyr2[yyn]);
// *****************************************************************************
#define tailSandwich(leftToken,rightToken,n, ...)                       \
  rhsTailSandwichVariadic(&yyval,yyn,n,leftToken,rightToken, __VA_ARGS__)

#define Yp1p(lhs, n1)                                                 \
  tailSandwich('(', ')',1,n1)

#define Yp3p(lhs, n1, n2, n3)                                         \
  tailSandwich('(', ')',yyr2[yyn],n1,n2,n3)

#define Yadrs(lhs,and,expr)                                           \
  astNode *i=astNewNode();                                            \
  i->tokenid=ADRS_IN;                                                 \
  astNode *o=astNewNode();                                            \
  o->tokenid=ADRS_OUT;                                                \
  RHS(lhs,i,and,expr,o)


// *****************************************************************************
// * Opérations
// *****************************************************************************

#define Yop3p(lhs, n1, op, n3)                                        \
  astNode *nOp=astNewNode();                                          \
  nOp->token=strdup(op2name(op->token));                              \
  nOp->tokenid=op->tokenid;                                           \
  astNode *pIn=astNewNode();                                          \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;              \
  astNode *nComa=astNewNode();                                        \
  nComa->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;          \
  astNode *pOut=astNewNode();                                         \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');             \
  RHS(lhs,nOp,pIn,n1,nComa,n3,pOut)
 
#define Zop3p(lhs, n1, op, n3)                                        \
  printf("\nyyn=%d, yyr1[yyn]=%d '%s' yytoken=%d %s\n",               \
         yyn,yyr1[yyn],yytname[yyr1[yyn]],yytoken,yytname[yytoken]);  \
  astNode *nOp=astNewNode();                                          \
  nOp->token=strdup(op2name(op->token));                              \
  nOp->tokenid=op->tokenid;                                           \
  astNode *pIn=astNewNode();                                          \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;              \
  astNode *nComa=astNewNode();                                        \
  nComa->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;          \
  astNode *pOut=astNewNode();                                         \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');             \
  RHS(lhs,nOp,pIn,n1,nComa,n3,pOut)

#define YopTernary5p(lhs, cond, question, ifState,doubleDot,elseState)  \
  astNode *nOp=astNewNode();                                            \
  nOp->token=strdup("opTernary");                                       \
  nOp->tokenid=0;                                                       \
  astNode *pIn=astNewNode();                                            \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;                \
  astNode *nComa=astNewNode();                                          \
  nComa->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;            \
  astNode *nComa2=astNewNode();                                         \
  nComa2->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;           \
  astNode *pOut=astNewNode();                                           \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');               \
  RHS(lhs,nOp,pIn,cond,nComa,ifState,nComa2,elseState,pOut)

#define YopDuaryExpression(lhs,ident,op,cond,ifState)                   \
  astNode *nOp=astNewNode();                                            \
  nOp->token=strdup("opTernary");                                       \
  nOp->tokenid=0;                                                       \
  astNode *pIn=astNewNode();                                            \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;                \
  astNode *nComa=astNewNode();                                          \
  nComa->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;            \
  astNode *nComa2=astNewNode();                                         \
  nComa2->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;           \
  astNode *pOut=astNewNode();                                           \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');               \
  astNode *elseState=astNewNodeRule(ident->rule,ident->ruleid);         \
  elseState->children=ident->children;                                  \
  RHS(lhs,ident,op,nOp,pIn,cond,nComa,ifState,nComa2,elseState,pOut)



// *****************************************************************************
// * Autres actions particulières
// *****************************************************************************

#define Ypow(lhs,n1,pow)                                              \
  astNode *pPow=astNewNode();                                         \
  pPow->token=strdup("pow");                                          \
  pPow->tokenid=IDENTIFIER;                                           \
  astNode *pIn=astNewNode();                                          \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(');               \
  astNode *pTwo=astNewNode();                                         \
  pTwo->token=strdup("," #pow ".0");pTwo->tokenid=IDENTIFIER;         \
  astNode *pOut=astNewNode();                                         \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');             \
  RHS(lhs,pPow,pIn,n1,pTwo,pOut)

#define remainY1(lhs)                                                 \
  astNode *timeRemainNode=astNewNode();                               \
  timeRemainNode->token=strdup("slurmTremain()");                     \
  timeRemainNode->tokenid=YYTRANSLATE(REMAIN);                        \
  RHS(lhs,timeRemainNode)

#define limitY1(lhs)                                                  \
  astNode *timeLimitNode=astNewNode();                                \
  timeLimitNode->token=strdup("slurmTlimit()");                       \
  timeLimitNode->tokenid=YYTRANSLATE(LIMIT);                          \
  RHS(lhs,timeLimitNode)

#define volatilePreciseY1(lhs, gmpType){                              \
    astNode *mpTypeNode=astNewNode();                                 \
    astNode *volatileNode=astNewNode();                               \
    if (gmpType==GMP_INTEGER)                                         \
      mpTypeNode->token=strdup("mpInteger");                          \
    else                                                              \
      mpTypeNode->token=strdup("mpReal");                             \
    volatileNode->tokenid=VOLATILE;                                   \
    volatileNode->token=strdup("VOLATILE");                           \
    mpTypeNode->tokenid=YYTRANSLATE(gmpType);                         \
    RHS(lhs,mpTypeNode,volatileNode);}

#define preciseY1(lhs, gmpType){                                      \
    astNode *mpTypeNode=astNewNode();                                 \
    if (gmpType==GMP_INTEGER)                                         \
      mpTypeNode->token=strdup("mpInteger");                          \
    else                                                              \
      mpTypeNode->token=strdup("mpReal");                             \
    mpTypeNode->tokenid=YYTRANSLATE(gmpType);                         \
    RHS(lhs,mpTypeNode);}

#define primeY1ident(lhs, ident)                                      \
  char token[1024];                                                   \
  astNode *mathlinkPrimeNode=astNewNode();                            \
  sprintf(token, "m_mathlink->Prime(%s);\n\t",ident->token);          \
  mathlinkPrimeNode->token=strdup(token);                             \
  mathlinkPrimeNode->tokenid=YYTRANSLATE(MATHLINK);                   \
  RHS(lhs,mathlinkPrimeNode)

#define primeY1(lhs, cst)                                             \
  char token[1024];                                                   \
  astNode *mathlinkPrimeNode=astNewNode();                            \
  sprintf(token, "m_mathlink->Prime(%s);\n\t",cst->token);            \
  mathlinkPrimeNode->token=strdup(token);                             \
  mathlinkPrimeNode->tokenid=YYTRANSLATE(MATHLINK);                   \
  RHS(lhs,mathlinkPrimeNode)


#endif // _NABLA_Y_H_
