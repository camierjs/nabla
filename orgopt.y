///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
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
%{
#include "nabla.h"
#include "nabla.y.h" 
#undef YYDEBUG
#define YYSTYPE astNode*
int yylineno;
char nabla_input_file[1024];
int yylex(void);
void yyerror(astNode **root, const char *s);
extern FILE *yyin;
%}
 
///////////////////////////////
// Terminals used in grammar //
///////////////////////////////
%token IDENTIFIER
%token ALPHA DELTA
%token TRUE FALSE
%token HEX_CONSTANT Z_CONSTANT R_CONSTANT

//////////////////////
// Specific options //
//////////////////////
%debug
%error-verbose
%start entry_point
%token-table
%parse-param {astNode **root}

%%

/////////////////////////////////
// Entry point of input stream //
/////////////////////////////////
entry_point: orgopt_inputstream{*root=$1;};
orgopt_inputstream
: orgopt_grammar {rhs;}
| orgopt_inputstream orgopt_grammar {astAddChild($1,$2);}
;

plus_moins:'+'|'-';

boolean
: TRUE {rhs;}
| FALSE {rhs;}
;

value
: HEX_CONSTANT
| Z_CONSTANT
| R_CONSTANT 
;

//id:IDENTIFIER {rhs;}

option
: '-' IDENTIFIER '=' boolean {rhs;}
| '-' IDENTIFIER '=' value {rhs;}
| '-' IDENTIFIER '=' plus_moins value {rhs;}
;

stars: '*'| stars '*';

header: stars IDENTIFIER;

///////////////
// ∇ grammar //
///////////////
orgopt_grammar: header | option;

%%

// ****************************************************************************
// * yyerror
// ****************************************************************************
void yyerror(astNode **root, const char *error){
  fflush(stdout);
  printf("\r%s:%d: %s\n",nabla_input_file,yylineno-1, error);
}


// ****************************************************************************
// * orgGrammar
// ****************************************************************************
static void orgGrammar(astNode * n){
  if (n->ruleid == rulenameToId("option")){
    const astNode *nnn=n->children->next->next->next;
    const char *ascii_id=utf2ascii(n->children->next->token);
    if (nnn->tokenid=='-' or nnn->tokenid=='+')
      printf(" --%s=%s%s",ascii_id,nnn->token,nnn->next->token);
    else
      if (nnn->ruleid == rulenameToId("boolean"))
        printf(" --%s=%s",ascii_id,nnn->children->token);
      else
        printf(" --%s=%s",ascii_id,nnn->token);
  }
  if(n->children != NULL) orgGrammar(n->children);
  if(n->next != NULL) orgGrammar(n->next);
}


// ****************************************************************************
// * main
// ****************************************************************************
int main(int argc, char * argv[]){
  astNode *root=NULL;
  if (argc<2) return printf("No input file!\n");
  printf("Input file is '%s'\n",argv[1]);
  
  const char *nabla_entity_name = "orgopt";
  if(!(yyin=fopen(argv[1],"r")))
    return NABLA_ERROR |
      dbg("\n[nablaParsing] Could not open '%s' file",
          argv[1]);
  dbg("\n[nablaParsing] Starting parsing");
  if (yyparse(&root)){
    fclose(yyin);
    return NABLA_ERROR | dbg("\n[nablaParsing] Error while parsing!");
  }
  fclose (yyin);
  astTreeSave(nabla_entity_name, root);
  orgGrammar(root);
  dbg("\n[nablaParsing] Closing & Quit");
  return 0;
}


// ****************************************************************************
// * rulenameToId
// ****************************************************************************
int rulenameToId(const char *rulename){
  unsigned int i;
  const size_t rnLength=strlen(rulename);
  for(i=0; yytname[i]!=NULL;i+=1){
    if (strlen(yytname[i])!=rnLength) continue;
    if (strcmp(yytname[i], rulename)!=0) continue;
    return i;
  }
  dbg("[rulenameToId] error with '%s'",rulename);
  return 1; // error
}


// *****************************************************************************
// * Standard rhsAdd
// *****************************************************************************
inline void rhsAdd(astNode **lhs,int yyn, astNode* *yyvsp){
  // Nombre d'éléments dans notre RHS
  const int yynrhs = yyr2[yyn];
  // On accroche le nouveau noeud au lhs
  astNode *first=*lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);
  // On va scruter tous les éléments
  // On commence par rajouter le premier comme fils
  astNode *next=yyvsp[(0+1)-(yynrhs)];
  astAddChild(first,next);
  // Et on rajoute des 'next' comme frères
  for(int yyi=1; yyi<yynrhs; yyi++){
    // On swap pour les frères
    first=next;
    next=yyvsp[(yyi+1)-(yynrhs)];
    astAddNext(first,next);
  }
}

