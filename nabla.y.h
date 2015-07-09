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
#ifndef _NABLA_Y_H_
#define _NABLA_Y_H_


// ****************************************************************************
// * Counting Variadic Number of Arguments
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
// * Right Hand Side of the Grammar
// ****************************************************************************
#define rhs rhsAdd(&yyval,yyn,yyvsp)
#define RHS(lhs, ...) rhsAddVariadic(&yyval,yyn,NB_ARGS(__VA_ARGS__),__VA_ARGS__)


// ****************************************************************************
// * Forall
// ****************************************************************************
#define forall rhsTailSandwich(&yyval,yyn,FORALL_INI,FORALL_END,yyvsp);
#define forallVariadic(lhs, ...)                                        \
  tailSandwichVariadic(FORALL_INI,FORALL_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)


// ****************************************************************************
// * Compound Jobs
// ****************************************************************************
#define compound_job(lhs,...)                                           \
  tailSandwich(COMPOUND_JOB_INI,COMPOUND_JOB_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)

#define compound_reduction(lhs,...)                                     \
  tailSandwich(COMPOUND_REDUCTION_INI,COMPOUND_REDUCTION_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)

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


// *****************************************************************************
// * Yadrs and YadrsSandwich that uses tailSandwich
// *****************************************************************************
#define Yadrs(lhs,and,expr)                                           \
  astNode *i=astNewNode();                                            \
  i->tokenid=ADRS_IN;                                                 \
  astNode *o=astNewNode();                                            \
  o->tokenid=ADRS_OUT;                                                \
  RHS(lhs,i,and,expr,o)
#define YadrsSandwich(lhs,nd,expr)                                    \
  tailSandwich(ADRS_IN,ADRS_OUT,2,nd,expr)


// *****************************************************************************
// * Operations Sandwich
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
// * Other singular operations
// *****************************************************************************
#define superNP1(lhs,ident)                                           \
  char *dest;                                                         \
  dest=malloc(1024);                                                        \
  dest=strcat(dest,ident->token);                                     \
  dest=strcat(dest,"np1");                                            \
  astNode *superNP1Node=astNewNode();                                 \
  superNP1Node->token=strdup(dest);                                   \
  superNP1Node->tokenid=IDENTIFIER;                                   \
  RHS(lhs,superNP1Node)

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
