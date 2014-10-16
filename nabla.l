/*****************************************************************************\
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : ncc.l     																	  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 07.01.2010																	  *
 * Updated  : 07.01.2010																	  *
 *****************************************************************************
 * Description: 																				  *
 * Supported UTF8 characters are:
 * αβγδεζηθικλμνξοπρςστυφχψω
 * ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
 *****************************************************************************
 * Date			Author	Description														  *
 * 07.01.2010	jscamier	Creation															  *
 *****************************************************************************/
%{
#include "nabla.h"
#include <stdio.h>
//#define YY_SKIP_YYWRAP
extern int yylineno;
#undef YY_NEED_STRLEN
#define YYSTYPE  astNode*   
#include "nabla.tab.h"
static int tok(YYSTYPE *yylval, int tokenid);
static void ignoreBlocComments(void); 
static void ignoreLineComments(void);
 
/* Terminals unused in grammar
"struct"				{ return tok(&yylval, STRUCT);}
"case"				{ return tok(&yylval, CASE);}
"default"			{ return tok(&yylval, DEFAULT);}
"enum"				{ return tok(&yylval, ENUM);}
"goto"				{ return tok(&yylval, GOTO);}
"switch"				{ return tok(&yylval, SWITCH);}
"union"				{ return tok(&yylval, UNION);}
"iNode"           { return tok(&yylval, INODE);}
"FILE"            { return tok(&yylval, FILETYPE);}
*/

%}

D					[0-9]
L					[a-zA-Z_αβγδεζηθικλμνξοπρςστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ𝜕]
H					[a-fA-F0-9]
E					[Ee][+-]?{D}+
FS					(f|F|l|L)
IS					(u|U|l|L)*
preprocs       [#][ \t]*[1-9][0-9]*[ \t]*\".*\"([ \t]*[1-3])*
includes	   	^[ ][i][n][c][l][u][d][e][ \t]*[\"<]*[a-zA-Z\./]*[\">]*

%%
"/*"					{ ignoreBlocComments();}
"//"              { ignoreLineComments();}
"auto"				{ return tok(&yylval, AUTO);}
"break"				{ return tok(&yylval, BREAK);}
"char"				{ return tok(&yylval, CHAR);}
"const"				{ return tok(&yylval, CONST);}
"continue"			{ return tok(&yylval, CONTINUE);}
"do"					{ return tok(&yylval, DO);}
"double"				{ return tok(&yylval, DOUBLE);}
"else"				{ return tok(&yylval, ELSE);}
"extern"				{ return tok(&yylval, EXTERN);}
"float"				{ return tok(&yylval, FLOAT);}
"for"					{ return tok(&yylval, FOR);}
"if"					{ return tok(&yylval, IF);}
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
"unsigned"			{ return tok(&yylval, UNSIGNED);}
"void"				{ return tok(&yylval, VOID);}
"volatile"			{ return tok(&yylval, VOLATILE);}
"while"				{ return tok(&yylval, WHILE);}

"#"			      { return tok(&yylval, DIESE);}
"@"			      { return tok(&yylval, AT);}
"foreach"			{ return tok(&yylval, FOREACH);}
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

"size_t"			   { return tok(&yylval, SIZE_T);}
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

"cartesian"       { return tok(&yylval, LIB_CARTESIAN);}
"xyz"             { return tok(&yylval, XYZ);}
"nextCell"        { return tok(&yylval, NEXTCELL);}
"prevCell"        { return tok(&yylval, PREVCELL);}
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

"mat"             { return tok(&yylval, MAT);}
"env"             { return tok(&yylval, ENV);}
"matenv"          { return tok(&yylval, LIB_MATENV);}
"material"        { return tok(&yylval, MATERIAL);}
"materials"       { return tok(&yylval, MATERIALS);}
"environments"    { return tok(&yylval, ENVIRONMENTS);}

"mathematica"     { return tok(&yylval, LIB_MATHEMATICA);}
"mathlink"        { return tok(&yylval, MATHLINK);}
"Prime"           { return tok(&yylval, PRIME);}

"gmp"             { return tok(&yylval, LIB_GMP);}
"precise"         { return tok(&yylval, GMP_PRECISE);}

"dft"             { return tok(&yylval, LIB_DFT);}

"mpi"             { return tok(&yylval, LIB_MPI);}
"reduce"          { return tok(&yylval, MPI_REDUCE);}

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

{L}({L}|{D})*		{ return tok(&yylval, IDENTIFIER);}
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
"\&\="		    		{ return tok(&yylval, AND_ASSIGN);}
"^="					{ return tok(&yylval, XOR_ASSIGN);}
"\|="					{ return tok(&yylval, IOR_ASSIGN);}

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

"[n]"             { return tok(&yylval, FOREACH_NODE_INDEX);}
"[c]"             { return tok(&yylval, FOREACH_CELL_INDEX);}


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
//"nand"             { return tok(&yylval, NAND_OP);}
//"⊼"					{ return tok(&yylval, NAND_OP);} // Nand 22BC
//"⊻"					{ return tok(&yylval, XOR_OP);} // Xor 22BB
//""					{ return tok(&yylval, XNOR_OP);}// Xnor F4A2
//"⊽"					{ return tok(&yylval, NOR_OP);} // Nor 22BD
//"::"					{ return tok(&yylval, NAMESPACE);}

// Absorbe line comments
static void ignoreLineComments(void){
  while(input()!='\n');
}

// Absorbe bloc comments
static void ignoreBlocComments(void){
	char c, c1;
loop:
	do {c = input();} while (c != '*' && c != 0);
	c1 = input();
	if (c1 != '/' && c != 0){
		unput(c1);
		goto loop;
	}
}

// Tokenizer
static int tok(YYSTYPE *yylval, int tokenid){
  //dbg("\n[astNewNode] line %d: '%s' (%d)", yylineno, yytext, tokenid);
  if (tokenid==PREPROCS){
    long line_jump=atol(yytext+1);
    assert(line_jump>0);
    //dbg("\n[astNewNode] preprocs: line_jump=%d and yylineno=%d", line_jump, yylineno);
    yylineno=line_jump-1;
  }
  (*yylval) = astNewNodeToken();
  (*yylval)->token=strdup(yytext);
  (*yylval)->tokenid=tokenid;
  return tokenid;
}
