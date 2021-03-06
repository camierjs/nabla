///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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


// ****************************************************************************
// * Forward declarations
// ****************************************************************************
void rhsAdd(node**,int,node**);
void rhsPatchAndAdd(const int, const char,node**,int,node**);
void rhsYSandwich(node**,int,node**,int,int);
void rhsTailSandwich(node**,int,int,int,node**);

void rhsAddVariadic(node**,int,int,...);
void rhsYSandwichVariadic(node**,int,int,int,int,...);
void rhsTailSandwichVariadic(node**,int,int,int,int,...);


// ****************************************************************************
// * Right Hand Side of the Grammar
// ****************************************************************************
#define rhs rhsAdd(&yyval,yyn,yyvsp)
#define rhsPatch(i,chr) rhsPatchAndAdd(i,chr,&yyval,yyn,yyvsp)
#define ast(...) rhsAddVariadic(&yyval,yyn,NB_ARGS(__VA_ARGS__),__VA_ARGS__)


// ****************************************************************************
// * Forall
// ****************************************************************************
#define forall rhsTailSandwich(&yyval,yyn,FORALL_INI,FORALL_END,yyvsp);
#define forallVariadic(lhs, ...)\
  tailSandwichVariadic(FORALL_INI,FORALL_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)


// ****************************************************************************
// * Compound Jobs
// ****************************************************************************
#define job rhsTailSandwich(&yyval,yyn,COMPOUND_JOB_INI,COMPOUND_JOB_END,yyvsp);

#define compound_job(lhs,...)\
  tailSandwich(COMPOUND_JOB_INI,COMPOUND_JOB_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)

#define compound_reduction(lhs,...)\
  tailSandwich(COMPOUND_REDUCTION_INI,COMPOUND_REDUCTION_END,NB_ARGS(__VA_ARGS__),__VA_ARGS__)

#define compound_job_without__NB_ARGS__(lhs,...)                       \
  assert(NB_ARGS(__VA_ARGS__)==yyr2[yyn]);\
  tailSandwich(COMPOUND_JOB_INI,COMPOUND_JOB_END,yyr2[yyn],__VA_ARGS__)


// *****************************************************************************
// * Tail Sandwich
// * printf("[1;33mYp2p=%d[m\n",yyr2[yyn]);
// *****************************************************************************
#define YopYop(leftToken,rightToken)\
  rhsYSandwich(&yyval,yyn,yyvsp,leftToken,rightToken)
#define YopYopVariadic(leftToken,rightToken, ...)\
  rhsYSandwichVariadic(&yyval,yyn,yyr2[yyn],leftToken,rightToken, __VA_ARGS__)

#define tailSandwich(leftToken,rightToken,n, ...)                       \
  rhsTailSandwichVariadic(&yyval,yyn,n,leftToken,rightToken, __VA_ARGS__)

#define Yp1p(lhs, n1)                                                 \
  tailSandwich('(', ')',1,n1)

#define Yp3p(lhs, n1, n2, n3)                                         \
  tailSandwich('(', ')',yyr2[yyn],n1,n2,n3)

// *****************************************************************************
// * Power Types
// *****************************************************************************
#define powerType(lhs,type,dims)                                       \
  ast(astNewNode("power",POWER),type,dims)

// *****************************************************************************
// * Yadrs and YadrsSandwich that uses tailSandwich
// *****************************************************************************
#define Yadrs(lhs,and,expr)                                           \
  ast(astNewNode(NULL,ADRS_IN),and,expr,astNewNode(NULL,ADRS_OUT))
#define YadrsSandwich(lhs,nd,expr)                                    \
  tailSandwich(ADRS_IN,ADRS_OUT,2,nd,expr)


// *****************************************************************************
// * Operations Sandwich
// *****************************************************************************
#define Yop3p(lhs, n1, op, n3)                                        \
  node *nOp=astNewNode(toolOpName(op->token_utf8),op->tokenid);    \
  node *pIn=astNewNode("(",YYTRANSLATE('('));                      \
  node *nComa=astNewNode(",",YYTRANSLATE(','));                    \
  node *pOut=astNewNode(")",YYTRANSLATE(')'));                     \
  ast(nOp,pIn,n1,nComa,n3,pOut)
 
#define YopTernary5p(lhs,cond,qstn,if,doubleDot,else)                   \
  node *nOp=astNewNode("opTernary",0);                               \
  node *pIn=astNewNode("(",YYTRANSLATE('('));                        \
  node *nComa=astNewNode(",",YYTRANSLATE(','));                      \
  node *nComa2=astNewNode(",",YYTRANSLATE(','));                     \
  node *pOut=astNewNode(")",YYTRANSLATE(')'));                       \
  ast(nOp,pIn,cond,nComa,if,nComa2,else,pOut)

#define YopDuaryExpression(lhs,ident,op,cond,ifState)                   \
  node *nOp=astNewNode("opTernary",0);                               \
  node *pIn=astNewNode("(",YYTRANSLATE('('));                        \
  node *nComa=astNewNode(",",YYTRANSLATE(','));                      \
  node *nComa2=astNewNode(",",YYTRANSLATE(','));                     \
  node *pOut=astNewNode(")",YYTRANSLATE(')'));                       \
  node *elseState=astNewNodeRule(ident->rule,ident->ruleid);         \
  elseState->children=ident->children;                                  \
  ast(ident,op,nOp,pIn,cond,nComa,ifState,nComa2,elseState,pOut)



// *****************************************************************************
// * Other singular operations
// * node *pi=astNewNode("(",YYTRANSLATE('('));                       
// * node *po=astNewNode(")",YYTRANSLATE(')'));                     
// * decl_spec->children=type_spec;                                      
// * type_spec->children=vd;                                             
// * node *decl_spec=astNewNode(NULL,rulenameToId("declaration_specifiers")); 
// *****************************************************************************
//#define voidNoIDvoid_rhs(lhs,at,cst,statement) voidIDvoid_rhs(lhs,"NoID",at,cst,statement)
//#define voidNoIDvoidIF_rhs(lhs,at,cst,ifs,pin,tst,pout,statement) voidIDvoid_rhs(lhs,"NoID",at,cst,statement)
#define voidIDvoidIF_rhs(lhs,id,at,cst,ifs,pin,tst,pout,statement) \
  node *decl_spec=astNewNodeRule("declaration_specifiers",rulenameToId("declaration_specifiers")); \
  node *type_spec=astNewNodeRule("type_specifier",rulenameToId("type_specifier")); \
  decl_spec->children=type_spec;                                        \
  node *rvd=astNewNode("void",VOID);                                    \
  node *avd=astNewNode("void",VOID);                                    \
  type_spec->children=rvd;                                              \
  node *decl=astNewNodeRule("declarator",rulenameToId("declarator"));   \
  node *direct_decl=astNewNodeRule("direct_declarator",rulenameToId("direct_declarator")); \
  node *direct_decl_bis=astNewNodeRule("direct_declarator",rulenameToId("direct_declarator")); \
  decl->children=direct_decl;                                           \
  direct_decl->children=id;                                             \
  direct_decl->children=direct_decl_bis;                                \
  direct_decl_bis->children=id;                                         \
  node *parameter_type_list=astNewNodeRule("parameter_type_list",rulenameToId("parameter_type_list")); \
  node *parameter_declaration=astNewNodeRule("parameter_declaration",rulenameToId("parameter_declaration")); \
  node *arg_type_spec=astNewNodeRule("type_specifier",rulenameToId("type_specifier")); \
  node *pi=astNewNode("(",YYTRANSLATE('('));                            \
  node *po=astNewNode(")",YYTRANSLATE(')'));                            \
  direct_decl_bis->next=pi;                                             \
  pi->next=parameter_type_list;                                         \
  parameter_type_list->next=po;                                         \
  parameter_type_list->children=parameter_declaration;                  \
  parameter_declaration->children=arg_type_spec;                        \
  arg_type_spec->children=avd;                                          \
  ast(decl_spec,decl,at,cst,ifs,pin,tst,pout,statement)
#define voidIDvoid_rhs(lhs,id,at,cst,statement)                         \
  node *decl_spec=astNewNodeRule("declaration_specifiers",rulenameToId("declaration_specifiers")); \
  node *type_spec=astNewNodeRule("type_specifier",rulenameToId("type_specifier")); \
  decl_spec->children=type_spec;                                        \
  node *rvd=astNewNode("void",VOID);                                    \
  node *avd=astNewNode("void",VOID);                                    \
  type_spec->children=rvd;                                              \
  node *decl=astNewNodeRule("declarator",rulenameToId("declarator"));   \
  node *direct_decl=astNewNodeRule("direct_declarator",rulenameToId("direct_declarator")); \
  node *direct_decl_bis=astNewNodeRule("direct_declarator",rulenameToId("direct_declarator")); \
  decl->children=direct_decl;                                           \
  direct_decl->children=id;                                             \
  direct_decl->children=direct_decl_bis;                                \
  direct_decl_bis->children=id;                                         \
  node *parameter_type_list=astNewNodeRule("parameter_type_list",rulenameToId("parameter_type_list")); \
  node *parameter_declaration=astNewNodeRule("parameter_declaration",rulenameToId("parameter_declaration")); \
  node *arg_type_spec=astNewNodeRule("type_specifier",rulenameToId("type_specifier")); \
  node *pi=astNewNode("(",YYTRANSLATE('('));                            \
  node *po=astNewNode(")",YYTRANSLATE(')'));                            \
  direct_decl_bis->next=pi;                                             \
  pi->next=parameter_type_list;                                         \
  parameter_type_list->next=po;                                         \
  parameter_type_list->children=parameter_declaration;                  \
  parameter_declaration->children=arg_type_spec;                        \
  arg_type_spec->children=avd;                                          \
  ast(decl_spec,decl,at,cst,statement)

#define nabla_job_id_rhs(lhs,prefix,id)                                 \
  node *decl_spec=astNewNodeRule("declaration_specifiers",rulenameToId("declaration_specifiers")); \
  node *type_spec=astNewNodeRule("type_specifier",rulenameToId("type_specifier")); \
  node *return_void=astNewNode("void",VOID);                         \
  node *pi=astNewNode("(",YYTRANSLATE('('));                         \
  node *po=astNewNode(")",YYTRANSLATE(')'));                         \
  node *parameter_type_list=astNewNodeRule("parameter_type_list",rulenameToId("parameter_type_list")); \
  node *parameter_list=astNewNodeRule("parameter_list",rulenameToId("parameter_list")); \
  node *parameter_declaration=astNewNodeRule("parameter_declaration",rulenameToId("parameter_declaration")); \
  node *parameter_type_spec=astNewNodeRule("type_specifier",rulenameToId("type_specifier")); \
  node *parameter_void=astNewNode("void",VOID);                      \
  decl_spec->children=type_spec;                                        \
  type_spec->children=return_void;                                      \
  parameter_type_list->children=parameter_list;                         \
  parameter_list->children=parameter_declaration;                       \
  parameter_declaration->children=parameter_type_spec;                  \
  parameter_type_spec->children=parameter_void;       \
  ast(prefix,decl_spec,id,pi,parameter_type_list,po)

#define superNP1(lhs,ident)                                           \
  char *dest=(char*)calloc(NABLA_MAX_FILE_NAME,sizeof(char));         \
  dest=strcat(dest,ident->token);                                     \
  dest=strcat(dest,"np1");                                            \
  node *superNP1Node=astNewNode(dest,IDENTIFIER);                  \
  ast(superNP1Node)

#define Ypow(lhs,n1,pow)                                              \
  node *pPow=astNewNode("pow",IDENTIFIER);                         \
  node *pIn=astNewNode("(",YYTRANSLATE('('));                      \
  node *pTwo=astNewNode("," #pow ".0",IDENTIFIER);                 \
  node *pOut=astNewNode(")",YYTRANSLATE(')'));                     \
  ast(pPow,pIn,n1,pTwo,pOut)

#define remainY1(lhs)                                                   \
  node *timeRemainNode=astNewNode("slurmTremain()",YYTRANSLATE(REMAIN)); \
  ast(timeRemainNode)

#define limitY1(lhs)                                                  \
  node *timeLimitNode=astNewNode("slurmTlimit()",YYTRANSLATE(LIMIT)); \
  ast(timeLimitNode)

#define volatilePreciseY1(lhs, gmpType){                              \
    node *mpTypeNode;                                              \
    if (gmpType==GMP_INTEGER)                                         \
      mpTypeNode=astNewNode("mpInteger",YYTRANSLATE(gmpType));        \
    else                                                              \
      mpTypeNode=astNewNode("mpReal",YYTRANSLATE(gmpType));           \
    node *volatileNode=astNewNode("VOLATILE",VOLATILE);            \
    ast(mpTypeNode,volatileNode);}

#define preciseY1(lhs, gmpType){                                      \
    node *mpTypeNode;                                              \
    if (gmpType==GMP_INTEGER)                                         \
      mpTypeNode=astNewNode("mpInteger",YYTRANSLATE(gmpType));        \
    else                                                              \
      mpTypeNode=astNewNode("mpReal",YYTRANSLATE(gmpType));           \
    ast(mpTypeNode);}

#define primeY1ident(lhs, ident)                                      \
  char token[1024];                                                   \
  sprintf(token, "m_mathlink->Prime(%s);\n\t",ident->token);          \
  node *mathlinkPrimeNode=astNewNode(token,YYTRANSLATE(MATHLINK);  \
  ast(mathlinkPrimeNode)

#define primeY1(lhs, cst)                                             \
  char token[1024];                                                   \
  sprintf(token, "m_mathlink->Prime(%s);\n\t",cst->token);            \
  node *mathlinkPrimeNode=astNewNode(token,YYTRANSLATE(MATHLINK)); \
  ast(mathlinkPrimeNode)


#endif // _NABLA_Y_H_
