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
#undef YY_BUF_SIZE
#define YY_BUF_SIZE 131072
//#undef YY_NEED_STRLEN
#define YYSTYPE  astNode*   
#include "orgopt.tab.h"
static int tok(YYSTYPE *yylval, int tokenid);
static void ignoreBlocComments(void); 
static void ignoreLineComments(void);
%}

D					[0-9]
L					[a-zA-Z_αβγδεζηθικλμνξοπρςστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ𝜕]
H					[a-fA-F0-9]
E					[Ee][+-]?{D}+
FS					(f|F|l|L)
IS					(u|U|l|L)*

%%

"/*"					{ ignoreBlocComments();}
"//"              { ignoreLineComments();}
"#+"              { ignoreLineComments();}

"true"				{ return tok(&yylval, TRUE);}
"false"				{ return tok(&yylval, FALSE);}

{L}({L}|{D})*		{ return tok(&yylval, IDENTIFIER); }

"*"			      { return tok(&yylval, '*');}
"-"			      { return tok(&yylval, '-');}
"+"			      { return tok(&yylval, '+');}
":noexport:"      { /*ignore*/ }
":ignore:"        { /*ignore*/ }

"α" {return tok(&yylval, ALPHA);}
"δ" {return tok(&yylval, DELTA);}
υ
0[x]{H}+{IS}?          { return tok(&yylval, HEX_CONSTANT);}
{D}+{IS}?			     { return tok(&yylval, Z_CONSTANT);}
L?'(\\.|[^\\'])+\'     { return tok(&yylval, R_CONSTANT);}
{D}+{E}{FS}?		     { return tok(&yylval, R_CONSTANT);}
{D}*"."{D}+({E})?{FS}? { return tok(&yylval, R_CONSTANT);}
{D}+"."{D}*({E})?{FS}? { return tok(&yylval, R_CONSTANT);}

"="					{ return tok(&yylval, '=');}

[ \t\v\f]		   { /*ignore*/ }

.						{ /* ignore bad characters */
  printf("\nREJECTing '%s' in ",yytext);
  REJECT;
}
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
  (*yylval) = astNewNode();
  (*yylval)->token=strdup(yytext);
  (*yylval)->tokenid=tokenid;
  return tokenid;
}

