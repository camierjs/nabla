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
void rhsAddGeneric(astNode**,int,int,...);
void rhsAddChildAndNexts(astNode**,int,astNode**);
void rhsTailSandwich(astNode**,int,int,int,int,...);

#define rhs rhsAddChildAndNexts(&yyval,yyn,yyvsp);
#define Y(lhs,...) rhsAddGeneric(&yyval,yyn,NB_ARGS(__VA_ARGS__),__VA_ARGS__)


// *****************************************************************************
// * Tail Sandwich
// * We still need these counting arguments macro,
// * because some of the rules are not comsuming all tokens
// * Perhaps they should not
// * compound_job_without__NB_ARGS__ is better
// * printf("[1;33mYp2p=%d[m\n",yyr2[yyn]);
// *****************************************************************************
#define tailSandwich(leftToken,rightToken,n, ...)                     \
  rhsTailSandwich(&yyval,yyn,n,leftToken,rightToken, __VA_ARGS__)

#define Yp1p(lhs, n1)                                                 \
  tailSandwich('(', ')',1,n1)

#define Yadrs(lhs,and,expr)                                           \
  astNode *i=astNewNodeToken();                                       \
  i->tokenid=ADRS_IN;                                                 \
  astNode *o=astNewNodeToken();                                       \
  o->tokenid=ADRS_OUT;                                                \
  Y(lhs,i,and,expr,o)

//tailSandwich(ADRS_IN, ADRS_IN,1,n1)

#define Yp3p(lhs, n1, n2, n3)                                         \
  tailSandwich('(', ')',yyr2[yyn],n1,n2,n3)

/*#define Yp4p(lhs, n1, n2, n3,n4)                                    \
  assert(4==yyr2[yyn]);                                               \
  tailSandwich('(', ')',yyr2[yyn],n1,n2,n3,n4)
*/
#define compound_job(lhs,...)                                           \
  tailSandwich(COMPOUND_JOB_INI,COMPOUND_JOB_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)

#define compound_job_without__NB_ARGS__(lhs,...)                        \
  assert(NB_ARGS(__VA_ARGS__)==yyr2[yyn]);                              \
  tailSandwich(COMPOUND_JOB_INI,COMPOUND_JOB_END,yyr2[yyn],__VA_ARGS__)

#define foreach(lhs,...)                                                \
  tailSandwich(FOREACH_INI,FOREACH_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)


// *****************************************************************************
// * Opérations
// *****************************************************************************

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
  Y(lhs,nOp,pIn,n1,nComa,n3,pOut)
 
#define Zop3p(lhs, n1, op, n3)                                          \
  printf("\nyyn=%d, yyr1[yyn]=%d '%s' yytoken=%d %s\n",   \
         yyn,yyr1[yyn],yytname[yyr1[yyn]],yytoken,yytname[yytoken]); \
  astNode *nOp=astNewNodeToken();                                       \
  nOp->token=strdup(op2name(op->token));                                \
  nOp->tokenid=op->tokenid;                                             \
  astNode *pIn=astNewNodeToken();                                       \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(') ;                \
  astNode *nComa=astNewNodeToken();                                     \
  nComa->token=strdup(",");nComa->tokenid=YYTRANSLATE(',') ;            \
  astNode *pOut=astNewNodeToken();                                      \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');               \
  Y(lhs,nOp,pIn,n1,nComa,n3,pOut)

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
  Y(lhs,nOp,pIn,cond,nComa,ifState,nComa2,elseState,pOut)

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
  Y(lhs,ident,op,nOp,pIn,cond,nComa,ifState,nComa2,elseState,pOut)



// *****************************************************************************
// * Autres actions particulières
// *****************************************************************************

#define Ypow(lhs,n1,pow)                                                \
  astNode *pPow=astNewNodeToken();                                      \
  pPow->token=strdup("pow");                                            \
  pPow->tokenid=IDENTIFIER;                                             \
  astNode *pIn=astNewNodeToken();                                       \
  pIn->token=strdup("(");pIn->tokenid=YYTRANSLATE('(');                 \
  astNode *pTwo=astNewNodeToken();                                      \
  pTwo->token=strdup("," #pow ".0");pTwo->tokenid=IDENTIFIER;           \
  astNode *pOut=astNewNodeToken();                                      \
  pOut->token=strdup(")");pOut->tokenid=YYTRANSLATE(')');               \
  Y(lhs,pPow,pIn,n1,pTwo,pOut)

#define remainY1(lhs)                                                 \
  astNode *timeRemainNode=astNewNodeToken();                          \
  timeRemainNode->token=strdup("slurmTremain()");                     \
  timeRemainNode->tokenid=YYTRANSLATE(REMAIN);                        \
  Y(lhs,timeRemainNode)

#define limitY1(lhs)                                                  \
  astNode *timeLimitNode=astNewNodeToken();                           \
  timeLimitNode->token=strdup("slurmTlimit()");                       \
  timeLimitNode->tokenid=YYTRANSLATE(LIMIT);                          \
  Y(lhs,timeLimitNode)

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
  Y(lhs,mpTypeNode,volatileNode);}

#define preciseY1(lhs, gmpType){                                      \
  astNode *mpTypeNode=astNewNodeToken();                              \
  if (gmpType==GMP_INTEGER)                                           \
    mpTypeNode->token=strdup("mpInteger");                            \
  else                                                                \
    mpTypeNode->token=strdup("mpReal");                               \
  mpTypeNode->tokenid=YYTRANSLATE(gmpType);                           \
  Y(lhs,mpTypeNode);}

#define primeY1ident(lhs, ident)                                      \
  char token[1024];                                                   \
  astNode *mathlinkPrimeNode=astNewNodeToken();                       \
  sprintf(token, "m_mathlink->Prime(%s);\n\t",ident->token);          \
  mathlinkPrimeNode->token=strdup(token);                             \
  mathlinkPrimeNode->tokenid=YYTRANSLATE(MATHLINK);                   \
  Y(lhs,mathlinkPrimeNode)

#define primeY1(lhs, cst)                                             \
  char token[1024];                                                   \
  astNode *mathlinkPrimeNode=astNewNodeToken();                       \
  sprintf(token, "m_mathlink->Prime(%s);\n\t",cst->token);            \
  mathlinkPrimeNode->token=strdup(token);                             \
  mathlinkPrimeNode->tokenid=YYTRANSLATE(MATHLINK);                   \
  Y(lhs,mathlinkPrimeNode)


#endif // _NABLA_Y_H_
