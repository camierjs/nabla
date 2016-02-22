%{
///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
// IDDN.FR.001.520002.000.S.P.2014.000.10500                                 //
//                                                                           //
// Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          //
//                  http://www.quut.com/c/ANSI-C-grammar-l-2011.html         //
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
#include "nabla.h"
#include <stdio.h>
#define YY_SKIP_YYWRAP
  int yywrap(void){
    // return 0; if we switch nabla input file
    return 1;
  }
  extern int yylineno;
  extern bool typedef_hit;
  extern bool direct_declarator_hit;
  extern nablaType *typedef_names;
  char *last_identifier=NULL;
#undef YY_NEED_STRLEN
#define YYSTYPE  astNode*   
#include "nabla.tab.h"
static int tok(YYSTYPE *yylval, int tokenid);
static void ignoreBlocComments(void); 
static void ignoreLineComments(void);
static int tok_type(YYSTYPE*);
%}

D					[0-9]
L					[a-zA-Z_αβγδεζηθικλμνξοπρςστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ𝜕]
H					[a-fA-F0-9]
E					[Ee][+-]?{D}+
FS					(f|F|l|L)
IS					(u|U|l|L)*
preprocs       [#][ \t]*[1-9][0-9]*[ \t]*\".*\"([ \t]*[1-4])*
includes	   	^[ ][i][n][c][l][u][d][e][ \t]*[\"<]*[a-zA-Z\./]*[\">]*

%%
"/*"					{ ignoreBlocComments();}
"//"              { ignoreLineComments();}
"auto"				{ return tok(&yylval, AUTO);}
"break"				{ return tok(&yylval, BREAK);}
"case"				{ return tok(&yylval, CASE);}
"char"				{ return tok(&yylval, CHAR);}
"const"				{ return tok(&yylval, CONST);}
"continue"			{ return tok(&yylval, CONTINUE);}
"default"			{ return tok(&yylval, DEFAULT);}
"do"					{ return tok(&yylval, DO);}
"double"				{ return tok(&yylval, DOUBLE);}
"else"				{ return tok(&yylval, ELSE);}
"enum"				{ return tok(&yylval, ENUM);}
"extern"				{ return tok(&yylval, EXTERN);}
"float"				{ return tok(&yylval, FLOAT);}
"for"					{ return tok(&yylval, FOR);}
"goto"				{ return tok(&yylval, GOTO);}
"if"					{ return tok(&yylval, IF);}
"is"					{ return tok(&yylval, IS);}
"inline" 			{ return tok(&yylval, INLINE);}
"int"					{ return tok(&yylval, INT);}
"long"				{ return tok(&yylval, LONG);}
"aligned"			{ return tok(&yylval, ALIGNED);}
"restrict"			{ return tok(&yylval, RESTRICT);}
"register"			{ return tok(&yylval, REGISTER);}
"return"				{ return tok(&yylval, RETURN);}
"short"				{ return tok(&yylval, SHORT);}
"signed"				{ return tok(&yylval, SIGNED);}
"sizeof"				{ return tok(&yylval, SIZEOF);}
"static"				{ return tok(&yylval, STATIC);}
"struct"				{ return tok(&yylval, STRUCT);}
"switch"				{ return tok(&yylval, SWITCH);}
"typedef"         { typedef_hit=true; return tok(&yylval, TYPEDEF);}
"union"				{ return tok(&yylval, UNION);}
"unsigned"			{ return tok(&yylval, UNSIGNED);}
"void"				{ return tok(&yylval, VOID);}
"volatile"			{ return tok(&yylval, VOLATILE);}
"while"				{ return tok(&yylval, WHILE);}
__attribute__     { return tok(&yylval, ATTRIBUTE);}
__restrict        { return tok(&yylval, RESTRICT);}
__asm__           { return tok(&yylval, ASM);}

"#"			      { return tok(&yylval, DIESE);}
"@"			      { return tok(&yylval, AT);}
"∀"               { return tok(&yylval, FORALL);}
"forall"	   		{ return tok(&yylval, FORALL);}
"foreach"			{ return tok(&yylval, FORALL);}
"all"					{ return tok(&yylval, ALL);}
"own"					{ return tok(&yylval, OWN);}
"outer"				{ return tok(&yylval, OUTER);}
"inner"				{ return tok(&yylval, INNER);}
"in"					{ return tok(&yylval, IN);}
"out"					{ return tok(&yylval, OUT);}
"inout"				{ return tok(&yylval, INOUT);}
"cell"				{ return tok(&yylval, CELL);}
"Cell"            { return tok(&yylval, CELLTYPE);}
"cells"				{ return tok(&yylval, CELLS);}
"face"				{ return tok(&yylval, FACE);}
"Face"            { return tok(&yylval, FACETYPE);}
"faces"				{ return tok(&yylval, FACES);}
"node"			   { return tok(&yylval, NODE);}
"Node"            { return tok(&yylval, NODETYPE);}
"nodes"			   { return tok(&yylval, NODES);}
"global"		   	{ return tok(&yylval, GLOBAL);}
"particle"			{ return tok(&yylval, PARTICLE);}
"Particle"        { return tok(&yylval, PARTICLETYPE);}
"particles"			{ return tok(&yylval, PARTICLES);}
"options"		   { return tok(&yylval, OPTIONS);}

"ℝ³⨯ℝ³"	         { return tok(&yylval, REAL3x3);}
"ℝ³"			      { return tok(&yylval, REAL3);}
"ℝ²"			      { return tok(&yylval, REAL2);}
"ℝ"			      { return tok(&yylval, REAL);}
"ℤ"		         { return tok(&yylval, INTEGER);}
"ℕ"		         { return tok(&yylval, NATURAL);}
"ℾ"		         { return tok(&yylval, BOOL);}

"size_t"			   { return tok(&yylval, SIZE_T);}
"bool"			   { return tok(&yylval, BOOL);}
"Bool"			   { return tok(&yylval, BOOL);}
"Real"			   { return tok(&yylval, REAL);}
"Real2"			   { return tok(&yylval, REAL2);}
"Real2x2"		   { return tok(&yylval, REAL2x2);}
"Real3"			   { return tok(&yylval, REAL3);}
"Real3x3"		   { return tok(&yylval, REAL3x3);}
"Integer"		   { return tok(&yylval, INTEGER);}
"Int32"		      { return tok(&yylval, INT32);}
"Int64"		      { return tok(&yylval, INT64);}
"Uid"             { return tok(&yylval, UIDTYPE);}
"file"            { return tok(&yylval, FILECALL);}
"File"            { return tok(&yylval, FILETYPE);}
"FILE"            { return tok(&yylval, FILETYPE);}
"ofstream"        { return tok(&yylval, OFSTREAM);}

"cartesian"       { return tok(&yylval, LIB_CARTESIAN);}
"xyz"             { return tok(&yylval, XYZ);}
"nextCell"        { return tok(&yylval, NEXTCELL);}
"prevCell"        { return tok(&yylval, PREVCELL);}
"nextCellX"        { return tok(&yylval, NEXTCELL_X);}
"nextCellY"        { return tok(&yylval, NEXTCELL_Y);}
"nextCellZ"        { return tok(&yylval, NEXTCELL_Z);}
"prevCellX"        { return tok(&yylval, PREVCELL_X);}
"prevCellY"        { return tok(&yylval, PREVCELL_Y);}
"prevCellZ"        { return tok(&yylval, PREVCELL_Z);}
"nextNode"        { return tok(&yylval, NEXTNODE);}
"prevNode"        { return tok(&yylval, PREVNODE);}
"previousLeft"    { return tok(&yylval, PREVLEFT);}
"previousRight"   { return tok(&yylval, PREVRIGHT);}
"nextLeft"        { return tok(&yylval, NEXTLEFT);}
"nextRight"       { return tok(&yylval, NEXTRIGHT);}
"backCell"        { return tok(&yylval, BACKCELL);}
"backCellUid"     { return tok(&yylval, BACKCELLUID);}
"frontCell"       { return tok(&yylval, FRONTCELL);}
"frontCellUid"    { return tok(&yylval, FRONTCELLUID);}

"mathematica"     { return tok(&yylval, LIB_MATHEMATICA);}
"mathlink"        { return tok(&yylval, MATHLINK);}
"Prime"           { return tok(&yylval, PRIME);}

"gmp"             { return tok(&yylval, LIB_GMP);}
"precise"         { return tok(&yylval, GMP_PRECISE);}

"dft"             { return tok(&yylval, LIB_DFT);}

"slurm"           { return tok(&yylval, LIB_SLURM);}

"ℵ"               { return tok(&yylval, LIB_ALEPH);}
"aleph"           { return tok(&yylval, LIB_ALEPH);}
"rhs"             { return tok(&yylval, ALEPH_RHS);}
"lhs"             { return tok(&yylval, ALEPH_LHS);}
"matrix"          { return tok(&yylval, ALEPH_MTX);}
"reset"           { return tok(&yylval, ALEPH_RESET);}
"solve"           { return tok(&yylval, ALEPH_SOLVE);}
"setValue"        { return tok(&yylval, ALEPH_SET);}
"getValue"        { return tok(&yylval, ALEPH_GET);}
"newValue"        { return tok(&yylval, ALEPH_NEW_VALUE);}
"addValue"        { return tok(&yylval, ALEPH_ADD_VALUE);}

"mail"            { return tok(&yylval, MAIL);}

"with"            { return tok(&yylval, WITH);}
"time"            { return tok(&yylval, TIME);}
"limit"           { return tok(&yylval, LIMIT);}
"remain"          { return tok(&yylval, REMAIN);}
"exit"            { return tok(&yylval, EXIT);}
"iteration"       { return tok(&yylval, ITERATION);}
"fatal"           { return tok(&yylval, FATAL);}
"nbNode"          { return tok(&yylval, NBNODE);}
"nbCell"          { return tok(&yylval, NBCELL);}
"this"            { return tok(&yylval, THIS);}
"lid" 	         { return tok(&yylval, LID);}
"sid" 	         { return tok(&yylval, SID);}
"uid" 	         { return tok(&yylval, UID);}
"boundaryCell"    { return tok(&yylval, BOUNDARY_CELL);}

"&&"					{ return tok(&yylval, AND_OP);}
"∧"					{ return tok(&yylval, AND_OP);}  // a788e2
"and"             { return tok(&yylval, AND_OP);}

"||"					{ return tok(&yylval, IOR_OP);}
"∨"					{ return tok(&yylval, IOR_OP);}  // a888e2
"or"              { return tok(&yylval, IOR_OP);}

{L}({L}|{D})*		{ return tok_type(&yylval); }
0[x]{H}+{IS}?          { return tok(&yylval, HEX_CONSTANT);}
0{D}+{IS}?             { return tok(&yylval, OCT_CONSTANT);}
{D}+{IS}?			     { return tok(&yylval, Z_CONSTANT);}
L?'(\\.|[^\\'])+\'     { return tok(&yylval, R_CONSTANT);}
{D}+{E}{FS}?		     { return tok(&yylval, R_CONSTANT);}
{D}*"."{D}+({E})?{FS}? { return tok(&yylval, R_CONSTANT);}
{D}+"."{D}*({E})?{FS}? { return tok(&yylval, R_CONSTANT);}
L?\"(\\.|[^\\\"])*\"	  { return tok(&yylval, STRING_LITERAL);}

"..."					{ return tok(&yylval, ELLIPSIS);}
">>="					{ return tok(&yylval, RSH_ASSIGN);}
"<<="					{ return tok(&yylval, LSH_ASSIGN);}
"+="					{ return tok(&yylval, ADD_ASSIGN);}
"-="					{ return tok(&yylval, SUB_ASSIGN);}
"*="					{ return tok(&yylval, MUL_ASSIGN);}
"/="					{ return tok(&yylval, DIV_ASSIGN);}
"%="					{ return tok(&yylval, MOD_ASSIGN);}
"\&\="		    	{ return tok(&yylval, AND_ASSIGN);}
"^="					{ return tok(&yylval, XOR_ASSIGN);}
"\|="					{ return tok(&yylval, IOR_ASSIGN);}
"<?="					{ return tok(&yylval, MIN_ASSIGN);}
">?="					{ return tok(&yylval, MAX_ASSIGN);}
"?="					{ return tok(&yylval, NULL_ASSIGN);}

">>"					{ return tok(&yylval, RSH_OP);}
"<<"					{ return tok(&yylval, LSH_OP);}
"++"					{ return tok(&yylval, INC_OP);}
"--"					{ return tok(&yylval, DEC_OP);}
"->"					{ return tok(&yylval, PTR_OP);}

"<="					{ return tok(&yylval, LEQ_OP);}
">="					{ return tok(&yylval, GEQ_OP);}
"=="					{ return tok(&yylval, EEQ_OP);}
"!="					{ return tok(&yylval, NEQ_OP);}

";"					{ return tok(&yylval, ';');}
"{"		      	{ return tok(&yylval, '{');}
"}"			      { return tok(&yylval, '}');}
","					{ return tok(&yylval, ',');}
":"					{ return tok(&yylval, ':');}
"="					{ return tok(&yylval, '=');}
"("					{ return tok(&yylval, '(');}
")"					{ return tok(&yylval, ')');}
"["		      	{ return tok(&yylval, '[');}
"]"			      { return tok(&yylval, ']');}
"."					{ return tok(&yylval, '.');}
"&"					{ return tok(&yylval, '&');}
"!"					{ return tok(&yylval, '!');}
"¬"					{ return tok(&yylval, '!');} // 00AC
"~"					{ return tok(&yylval, '~');}
"-"					{ return tok(&yylval, '-');}
"+"					{ return tok(&yylval, '+');}
"*"					{ return tok(&yylval, '*');}
"/"					{ return tok(&yylval, '/');}
"%"					{ return tok(&yylval, '%');}
"<"					{ return tok(&yylval, '<');}
">"					{ return tok(&yylval, '>');}
"^"					{ return tok(&yylval, '^');}
"|"					{ return tok(&yylval, '|');}
"?"					{ return tok(&yylval, '?');}

"²"					{ return tok(&yylval, SUPERSCRIPT_DIGIT_TWO);}
"³"					{ return tok(&yylval, SUPERSCRIPT_DIGIT_THREE);}
"√"					{ return tok(&yylval, SQUARE_ROOT_OP);}// SQUARE ROOT 9a88e2
"∛"					{ return tok(&yylval, CUBE_ROOT_OP);}//9a88e2

"½"					{ return tok(&yylval, FRACTION_ONE_HALF_CST);}//bdc2
"⅓"					{ return tok(&yylval, FRACTION_ONE_THIRD_CST);}//9385e2
"¼"					{ return tok(&yylval, FRACTION_ONE_QUARTER_CST);}//bcc2
"⅛"					{ return tok(&yylval, FRACTION_ONE_EIGHTH_CST);}//9b85e2

"∞"               { return tok(&yylval, BUILTIN_INFF);}//9e88e2

"ⁿ⁺¹"             { return tok(&yylval, SUPERSCRIPT_N_PLUS_ONE);}

"[n]"             { return tok(&yylval, FORALL_NODE_INDEX);}
"[c]"             { return tok(&yylval, FORALL_CELL_INDEX);}
"[m]"             { return tok(&yylval, FORALL_MTRL_INDEX);}


"⋅"					{ return tok(&yylval, CENTER_DOT_OP);} // MIDDLE DOT 858be2
"⨯"					{ return tok(&yylval, CROSS_OP);}//afa8e2
"⤫"					{ return tok(&yylval, CROSS_OP_2D);}//afa8e2
"⊗"					{ return tok(&yylval, CIRCLED_TIMES_OP);} // 978ae2
"⨂"					{ return tok(&yylval, N_ARY_CIRCLED_TIMES_OP);}//82a8e2
"⊛"					{ return tok(&yylval, CIRCLED_ASTERISK_OP);}//9b8ae2

[ \t\v\f]		   { /*ignore*/ }

{preprocs}		   { return tok(&yylval, PREPROCS); }
{includes}		   { return tok(&yylval, INCLUDES); }
.						{ /* ignore bad characters */
  printf("\nREJECTing '%s' in ",yytext);
  REJECT;}
\n                { yylineno+=1;}
%% 

// ****************************************************************************
// * Get rid of line comments
// ****************************************************************************
static void ignoreLineComments(void){
  while(input()!='\n');
  yylineno+=1;
}

// ****************************************************************************
// * Get rid of bloc comments
// ****************************************************************************
static void ignoreBlocComments(void){
	char c, c1;
loop:
	do {c = input(); if (c=='\n') yylineno+=1; } while (c != '*' && c != 0);
	c1 = input();
	if (c1 != '/' && c != 0){
		unput(c1);
		goto loop;
	}
}

// ****************************************************************************
// * Tokenizer
// ****************************************************************************
static int tok(YYSTYPE *yylval, int tokenid){
  //dbg("\n[astNewNode] line %d: '%s' (%d)", yylineno, yytext, tokenid);
  if (tokenid==PREPROCS){
    long line_jump=atol(yytext+1);
    assert(line_jump>0);
    //dbg("\n[astNewNode] preprocs: line_jump=%d and yylineno=%d", line_jump, yylineno);
    yylineno=line_jump-1;
  }
  (*yylval) = astNewNode();
  (*yylval)->token=strdup(yytext);
  (*yylval)->tokenid=tokenid;
  return tokenid;
}


// ****************************************************************************
// *
// ****************************************************************************
static int tok_type(YYSTYPE *yylval){
  (*yylval) = astNewNode();
  (*yylval)->token=strdup(yytext);
  last_identifier=strdup(yytext);
  //dbg("\n\t[tok_type] last_identifier is %s!",yytext);
  // On check s'il est déjà connu comme TYPEDEF_NAME
  if (nMiddleTypeFindName(typedef_names,yytext)!=NULL){
    dbg("\n\t[tok_type] TYPEDEF_NAME: %s <=> known type!",yytext);
    (*yylval)->tokenid=TYPEDEF_NAME;
    return TYPEDEF_NAME;
  }
  if (strstr(yytext,"__builtin")!=NULL){
    dbg("\n\t[tok_type] BUILTIN TYPE: %s!",yytext);
    (*yylval)->tokenid=TYPEDEF_NAME;
    return TYPEDEF_NAME;
  }
  (*yylval)->tokenid=IDENTIFIER;
  return IDENTIFIER;
}
