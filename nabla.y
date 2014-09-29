/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : ncc.y     																	  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 07.01.2010																	  *
 * Updated  : 12.11.2012																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 07.01.2010	jscamier	Creation															  *
 * 12.11.2012	jscamier	Incorporation Arcane Framework    						  *
 *****************************************************************************/
%{
#include "nabla.h"
#include <stdio.h>
#include <stdlib.h>
  
#define YYSTYPE astNode*

int yylex(void);
void yyerror(astNode **root, char *s);
 
bool type_volatile=false;
bool type_precise=false;
%}

 
/////////////////////////////////
// Terminals unused in grammar //
/////////////////////////////////
/*
  %token COMMENTS SINGLE_LINE_COMMENTS
  %token NOPINCLUDES
  %token NAMESPACE
  %token STRUCT UNION ENUM
  %token CASE DEFAULT GOTO
  %token SWITCH
  %token NAND_OP NOR_OP XOR_OP XNOR_OP
  %token INODE
*/

 // C-GRAMMAR
%token SPACE PREPROCS INCLUDES 
%token IDENTIFIER STRING_LITERAL QUOTE_LITERAL SIZEOF
%token PTR_OP INC_OP DEC_OP LSH_OP RSH_OP LEQ_OP GEQ_OP EEQ_OP NEQ_OP
%token AND_OP IOR_OP MUL_ASSIGN DIV_ASSIGN MOD_ASSIGN ADD_ASSIGN
%token SUB_ASSIGN LSH_ASSIGN RSH_ASSIGN AND_ASSIGN
%token XOR_ASSIGN IOR_ASSIGN
%token EXTERN STATIC AUTO REGISTER RESTRICT ALIGNED
%token CHAR SHORT INT LONG SIGNED UNSIGNED FLOAT DOUBLE CONST VOLATILE VOID
%token ELLIPSIS
%token IF ELSE WHILE DO FOR CONTINUE BREAK RETURN
%token INLINE GLOBAL
%token HEX_CONSTANT OCT_CONSTANT Z_CONSTANT R_CONSTANT
%token CALL ARGS POSTFIX_CONSTANT POSTFIX_CONSTANT_VALUE
%token PREFIX_PRIMARY_CONSTANT POSTFIX_PRIMARY_CONSTANT

 // MATHS tokens
%token SQUARE_ROOT_OP CUBE_ROOT_OP N_ARY_CIRCLED_TIMES_OP
%token CENTER_DOT_OP CROSS_OP CROSS_OP_2D CIRCLED_TIMES_OP CIRCLED_ASTERISK_OP
%token FRACTION_ONE_HALF_CST FRACTION_ONE_THIRD_CST
%token FRACTION_ONE_QUARTER_CST FRACTION_ONE_EIGHTH_CST
%token BUILTIN_INFF
%token SUPERSCRIPT_DIGIT_TWO SUPERSCRIPT_DIGIT_THREE

 // SPECIFIC NABLA GRAMMAR
%token COMPOUND_JOB_INI
%token COMPOUND_JOB_END
%token COORDS OPTIONS
%token AT DIESE
%token IN OUT INOUT
%token ALL OWN INNER OUTER
%token BOOL INTEGER INT32 INT64 REAL REAL2 REAL2x2 REAL3 REAL3x3 UIDTYPE SIZE_T
%token CELLTYPE NODETYPE FACETYPE
%token CELL CELLS FACE FACES NODE NODES
%token FOREACH FOREACH_INI FOREACH_END FOREACH_NODE_INDEX FOREACH_CELL_INDEX
%token PARTICLE PARTICLES PARTICLETYPE
%token FILETYPE

 // Nabla Cartesian
%token XYZ NEXTCELL PREVCELL NEXTNODE PREVNODE PREVLEFT PREVRIGHT NEXTLEFT NEXTRIGHT

 // Nabla Materials
%token MAT MATERIAL MATERIALS ENV ENVIRONMENT ENVIRONMENTS

 // Nabla LIBRARIES
%token LIB_MPI LIB_ALEPH LIB_CARTESIAN LIB_MATENV LIB_GMP LIB_MATHEMATICA LIB_SLURM MAIL LIB_MAIL LIB_DFT

 // ALEPH tokens
%token ALEPH_RHS ALEPH_LHS ALEPH_MTX ALEPH_RESET ALEPH_SOLVE ALEPH_SET ALEPH_GET ALEPH_NEW_VALUE ALEPH_ADD_VALUE

 // MPI tokens
%token MPI_REDUCE

 // GMP tokens 
%token GMP_PRECISE GMP_INTEGER GMP_REAL

 // SLURM tokens
%token REMAIN LIMIT

 // Mathematica tokens
%token MATHLINK PRIME

 // SYSTEM Nabla
%token LID SID UID THIS NBCELL NBNODE FATAL
%token BACKCELL BACKCELLUID FRONTCELL FRONTCELLUID
%token BOUNDARY_CELL
%token WITH
%token TIME EXIT ITERATION

 // If-Else Ambiguity
%nonassoc REMOVE_SHIFT_REDUCE_MESSAGE_OF_IF_ELSE_AMBIGUITY
%nonassoc ELSE

 // Specific options
%debug
%error-verbose
%start entry_point
%token-table
%parse-param {astNode **root}
%right '?' ':' ','

%%

/////////////////////////////////
// Entry point of input stream //
/////////////////////////////////
entry_point: nabla_inputstream{*root=$1;};
nabla_inputstream: nabla_grammar {Y1($$,$1)}
| nabla_inputstream nabla_grammar {astAddChild($1,$2);};


///////////////////////////
// ∇ scopes: std & std+@ //
///////////////////////////
start_scope: '{' {Y1($$,$1)};
end_scope: '}' {Y1($$,$1)} | '}' AT at_constant {Y3($$,$1,$2,$3)};


//////////////////////////////////////////////////
// ∇ types, qualifiers, specifier, lists & name //
//////////////////////////////////////////////////
type_specifier
: VOID {Y1($$,$1)}
| CHAR {Y1($$,$1)} 
| SHORT {Y1($$,$1)}
| INT {Y1($$,$1)}
| LONG {Y1($$,$1)}
| FLOAT {Y1($$,$1)}
| DOUBLE {Y1($$,$1)}
| SIGNED {Y1($$,$1)}
| UNSIGNED {Y1($$,$1)}
| REAL2 {Y1($$,$1)}
| REAL3 {Y1($$,$1)}
| REAL3x3 {Y1($$,$1)}
| REAL2x2 {Y1($$,$1)}
| BOOL {Y1($$,$1)}
| SIZE_T {Y1($$,$1)}
//| REAL {Y1($$,$1)}
//| INTEGER {Y1($$,$1)}
| REAL { if (type_precise) preciseY1($$,GMP_REAL) else Y1($$,$1); type_precise=type_volatile=false;}
| INTEGER {
    if (type_precise){
      if (type_volatile) volatilePreciseY1($$,GMP_INTEGER)
      else preciseY1($$,GMP_INTEGER)
    }else Y1($$,$1);
    type_precise=type_volatile=false;
  }
| INT32 {Y1($$,$1)}
| INT64 {Y1($$,$1)} 
| CELLTYPE {Y1($$,$1)}
| NODETYPE {Y1($$,$1)}
| PARTICLETYPE {Y1($$,$1)}
| FACETYPE {Y1($$,$1)}
| UIDTYPE {Y1($$,$1)}
| FILETYPE {Y1($$,$1)} 
//| MATERIAL {Y1($$,$1)}
//!| struct_or_union_specifier {Y1($$,$1)}
//!| enum_specifier {Y1($$,$1)}
; 
storage_class_specifier
: EXTERN {Y1($$,$1)}
| STATIC {Y1($$,$1)}
| AUTO {Y1($$,$1)}
| INLINE {Y1($$,$1)}
| REGISTER {Y1($$,$1)}
;
type_qualifier
: CONST {Y1($$,$1)}
| ALIGNED {Y1($$,$1)}
| VOLATILE {Y1($$,$1);type_volatile=true;}
| GMP_PRECISE {Y1($$,$1);type_precise=true;}
;
type_qualifier_list:
  type_qualifier {Y1($$,$1)}
| type_qualifier_list type_qualifier {Y2($$,$1,$2)}
;
specifier_qualifier_list
: type_specifier {Y1($$,$1)}
| type_specifier specifier_qualifier_list{Y2($$,$1,$2)}
| type_qualifier {Y1($$,$1)}
| type_qualifier specifier_qualifier_list{Y2($$,$1,$2)}
;
type_name
: specifier_qualifier_list {Y1($$,$1)}
| specifier_qualifier_list abstract_declarator{Y2($$,$1,$2)}
;


///////////////////////////////////////////////////////////
// ∇ item(s), group, region, family & system definitions //
///////////////////////////////////////////////////////////
nabla_matenv: MATERIAL {Y1($$,$1)}| ENVIRONMENT {Y1($$,$1)};
nabla_matenvs: MATERIALS {Y1($$,$1)} | ENVIRONMENTS {Y1($$,$1)};
nabla_item
: CELL {Y1($$,$1)}
| NODE {Y1($$,$1)}
| FACE {Y1($$,$1)}
| PARTICLE {Y1($$,$1)}
;
nabla_items
: CELLS {Y1($$,$1)}
| NODES {Y1($$,$1)}
| FACES {Y1($$,$1)}
| GLOBAL {Y1($$,$1)}
| PARTICLES {Y1($$,$1)}
;
nabla_scope: OWN {Y1($$,$1)} | ALL {Y1($$,$1)};
nabla_region: INNER {Y1($$,$1)} | OUTER {Y1($$,$1)};
nabla_family
: nabla_items {Y1($$,$1)}
| nabla_matenvs {Y1($$,$1)}
| nabla_scope nabla_items {Y2($$,$1,$2)}
| nabla_region nabla_items {Y2($$,$1,$2)}
| nabla_scope nabla_region nabla_items {Y3($$,$1,$2,$3)}
;
nabla_system
: LID {Y1($$,$1)}
| SID {Y1($$,$1)}
| UID {Y1($$,$1)}
| THIS{Y1($$,$1)}
| NBNODE{Y1($$,$1)}
| NBCELL{Y1($$,$1)}
//| FATAL{Y1($$,$1)}
| BOUNDARY_CELL{Y1($$,$1)}
| BACKCELL{Y1($$,$1)}
| BACKCELLUID{Y1($$,$1)}
| FRONTCELL{Y1($$,$1)}
| FRONTCELLUID{Y1($$,$1)}
| NEXTCELL {Y1($$,$1)}
| PREVCELL {Y1($$,$1)}
| NEXTNODE {Y1($$,$1)}
| PREVNODE {Y1($$,$1)}
| PREVLEFT {Y1($$,$1)}
| PREVRIGHT {Y1($$,$1)}
| NEXTLEFT {Y1($$,$1)}
| NEXTRIGHT {Y1($$,$1)}
| TIME {Y1($$,$1)}
| TIME REMAIN {remainY1($$)}
| TIME LIMIT {limitY1($$)}
| EXIT {Y1($$,$1)}
| ITERATION {Y1($$,$1)}
| MAIL {Y1($$,$1)}
;


//////////////
// Pointers //
//////////////
pointer:
  '*' {Y1($$,$1)}
| '*' type_qualifier_list{Y2($$,$1,$2)}
| '*' RESTRICT {Y2($$,$1,$2)}
| '*' type_qualifier_list pointer{Y2($$,$1,$2)}
;

//////////////////
// INITIALIZERS //
//////////////////
initializer:
  assignment_expression {Y1($$,$1)}
| '{' initializer_list '}'{Y3($$,$1,$2,$3)}
| '{' initializer_list ',' '}'{Y4($$,$1,$2,$3,$4)}
;
initializer_list
: initializer {Y1($$,$1)}
| initializer_list ',' initializer {Y3($$,$1,$2,$3)}
;


//////////////////
// DeclaraTIONS //
//////////////////
declaration_specifiers
: storage_class_specifier {Y1($$,$1)}
| storage_class_specifier declaration_specifiers{Y2($$,$1,$2)}
| type_specifier {Y1($$,$1)}
| type_specifier declaration_specifiers{Y2($$,$1,$2)}
| type_qualifier {Y1($$,$1)}
| type_qualifier declaration_specifiers{Y2($$,$1,$2)}
;
declaration
: PREPROCS {Y1($$,$1)} // On peut avoir des preprocs ici
//| NOPINCLUDES {Y1($$,$1)}
// On patche l'espace qui nous a été laissé par le sed pour remettre le bon #include
| INCLUDES {$1->token[0]='#';Y1($$,$1)}
| declaration_specifiers ';'{Y1($$,$1)}
| declaration_specifiers init_declarator_list ';' {Y3($$,$1,$2,$3)}  
;
declaration_list
:	declaration{Y1($$,$1)}
|	declaration_list declaration{Y2($$,$1,$2)}
;


/////////////////
// DeclaraTORS //
/////////////////
declarator
: pointer direct_declarator{Y2($$,$1,$2)}
| direct_declarator {Y1($$,$1)}
;
// Identifier list for direct declarators
identifier_list
: IDENTIFIER {Y1($$,$1)}
| identifier_list ',' IDENTIFIER	{Y3($$,$1,$2,$3)}
;
direct_declarator
: IDENTIFIER {Y1($$,$1)}
| '(' declarator ')'{Y3($$,$1,$2,$3)}
| direct_declarator '[' constant_expression ']'{Y4($$,$1,$2,$3,$4)}
| direct_declarator '[' ']'{Y3($$,$1,$2,$3)}
| direct_declarator '(' parameter_type_list ')'{Y4($$,$1,$2,$3,$4)}
| direct_declarator '(' identifier_list ')'{Y4($$,$1,$2,$3,$4)}
| direct_declarator '(' ')'{Y3($$,$1,$2,$3)}
;
init_declarator
:	declarator {Y1($$,$1)}
// Permet de faire des appels constructeurs à-là '=Real3(0.0,0.0,0.0)'
//|	declarator '=' type_specifier initializer{Y4($$,$1,$2,$3,$4)}
|	declarator '=' type_specifier '(' ')' {Y4($$,$1,$2,$3,$4)}
|	declarator '=' initializer{Y3($$,$1,$2,$3)}
;
init_declarator_list
:	init_declarator {Y1($$,$1)}
|	init_declarator_list ',' init_declarator{Y3($$,$1,$2,$3)}
;
abstract_declarator
: 	pointer {Y1($$,$1)}
|	direct_abstract_declarator {Y1($$,$1)}
|	pointer direct_abstract_declarator{Y2($$,$1,$2)}
;
direct_abstract_declarator
: '(' abstract_declarator ')'{Y3($$,$1,$2,$3)}
| '[' ']'{Y2($$,$1,$2)}
| '[' constant_expression ']'{Y3($$,$1,$2,$3)}
| direct_abstract_declarator '[' ']'{Y3($$,$1,$2,$3)}
| direct_abstract_declarator '[' constant_expression ']'{Y4($$,$1,$2,$3,$4)}
| '(' ')'{Y2($$,$1,$2)}
| '(' parameter_type_list ')'{Y3($$,$1,$2,$3)}
| direct_abstract_declarator '(' ')'{Y3($$,$1,$2,$3)}
| direct_abstract_declarator '(' parameter_type_list ')'{Y4($$,$1,$2,$3,$4)}
;


////////////////////
// Std parameters //
////////////////////
parameter_type_list
:	parameter_list {Y1($$,$1)}
|	parameter_list ',' ELLIPSIS{Y3($$,$1,$2,$3)}
;
parameter_list
:	parameter_declaration {Y1($$,$1)}
|	parameter_list ',' parameter_declaration{Y3($$,$1,$2,$3)}
;
parameter_declaration
: nabla_xyz_declaration {Y1($$,$1)}
| nabla_mat_declaration {Y1($$,$1)}
| nabla_env_declaration {Y1($$,$1)}
| declaration_specifiers declarator {Y2($$,$1,$2)}
| declaration_specifiers abstract_declarator {Y2($$,$1,$2)}
| declaration_specifiers {Y1($$,$1)}
;


//////////////////////////////
// ∇ xyz/mat/env parameters //
//////////////////////////////
nabla_xyz_direction:IDENTIFIER {Y1($$,$1)};
nabla_xyz_declaration: XYZ nabla_xyz_direction {Y2($$,$1,$2)};
nabla_mat_material:IDENTIFIER {Y1($$,$1)};
nabla_mat_declaration: MAT nabla_mat_material {Y2($$,$1,$2)};
nabla_env_environment:IDENTIFIER {Y1($$,$1)};
nabla_env_declaration: ENV nabla_env_environment {Y2($$,$1,$2)};
nabla_parameter_declaration
: nabla_item direct_declarator {Y2($$,$1,$2)};
nabla_parameter_list
: nabla_parameter_declaration {Y1($$,$1)}
| nabla_parameter_list ',' nabla_parameter_declaration {Y2($$,$1,$3)};

// !? WTF between these two 'nabla_parameter_list' ?!

/////////////////////////
// ∇ IN/OUT parameters //
/////////////////////////
nabla_parameter
: nabla_in_parameter_list {Y1($$,$1)}
| nabla_out_parameter_list {Y1($$,$1)}
| nabla_inout_parameter_list {Y1($$,$1)};
nabla_parameter_list
: nabla_parameter {Y1($$,$1)}
| nabla_parameter_list nabla_parameter{Y2($$,$1,$2)};
nabla_in_parameter_list: IN '(' nabla_parameter_list ')' {Y2($$,$1,$3)}; 
nabla_out_parameter_list: OUT '(' nabla_parameter_list ')' {Y2($$,$1,$3)};  
nabla_inout_parameter_list: INOUT '(' nabla_parameter_list ')' {Y2($$,$1,$3)};  


/////////////////
// EXPRESSIONS //
//  - primary
//  - postfix
//  - unary
/////////////////
primary_expression
: BUILTIN_INFF {Y1($$,$1)}
| FRACTION_ONE_HALF_CST {Y1($$,$1)}
| FRACTION_ONE_THIRD_CST {Y1($$,$1)}
| FRACTION_ONE_QUARTER_CST {Y1($$,$1)}
| FRACTION_ONE_EIGHTH_CST {Y1($$,$1)}
| DIESE {Y1($$,$1)}  // Permet d'écrire un '#' à la place d'un [c|n]
| IDENTIFIER {Y1($$,$1)}
//!| NAMESPACE IDENTIFIER {Y2($$,$1,$2)}
//| IDENTIFIER '<' REAL '>' NAMESPACE IDENTIFIER {Y1($$,$6)}
//!| IDENTIFIER NAMESPACE IDENTIFIER {Y3($$,$1,$2,$3)}
| nabla_item {Y1($$,$1)} // Permet de rajouter les items Nabla au sein des corps de fonctions
| nabla_system {Y1($$,$1)}
| HEX_CONSTANT {Y1($$,$1)} 
| OCT_CONSTANT {Y1($$,$1)}
| Z_CONSTANT {Y1($$,$1)}
| R_CONSTANT {Y1($$,$1)}
| QUOTE_LITERAL {Y1($$,$1)}
| STRING_LITERAL {Y1($$,$1)}
| '(' expression ')'	{Y3($$,$1,$2,$3)}
;
argument_expression_list
: assignment_expression {Y1($$,$1)}
| argument_expression_list ',' assignment_expression {Y3($$,$1,$2,$3)}
;
postfix_expression
: primary_expression {Y1($$,$1)} 
| postfix_expression FOREACH_NODE_INDEX {Y2($$,$1,$2)}
| postfix_expression FOREACH_CELL_INDEX {Y2($$,$1,$2)}
| postfix_expression '[' expression ']' {Y4($$,$1,$2,$3,$4)}
//| type_specifier '(' ')'  {Y3($$,$1,$2,$3)}
//type_specifier '(' expression_list ')'  {Y4($$,$1,$2,$3,$4)}
//| postfix_expression '[' nabla_system ']' {Y4($$,$1,$2,$3,$4)}
//| postfix_expression '[' Z_CONSTANT ']'  {
//  // On rajoute un noeud pour annoncer qu'il faut peut-être faire quelque chose lors de ce postfix
//  astNode *cstPostNode=astNewNodeToken();
//  cstPostNode->tokenid=POSTFIX_CONSTANT;
//  astNode *cstValueNode=astNewNodeToken();
//  cstValueNode->tokenid=POSTFIX_CONSTANT_VALUE;
//  Y6($$,cstPostNode,$1,$2,$3,cstValueNode,$4)
// }
| postfix_expression '(' ')' {Y3($$,$1,$2,$3)}
// On traite l'appel à fatal différemment qu'un CALL standard
| FATAL '(' argument_expression_list ')' {Y4($$,$1,$2,$3,$4)}
| postfix_expression '(' argument_expression_list ')'{
  // On rajoute un noeud pour annoncer qu'il faut peut-être faire quelque chose lors de l'appel à la fonction
  astNode *callNode=astNewNodeToken();
  // On DOIT laisser un token != NULL!
  callNode->token=strdup("");///*call*/");
  callNode->tokenid=CALL;
  astNode *argsNode=astNewNodeToken();
  argsNode->token=strdup("");///*args*/");
  argsNode->tokenid=ARGS;
  Y6($$,callNode,$1,$2,$3,argsNode,$4)
    }
| postfix_expression '.' IDENTIFIER {Y3($$,$1,$2,$3)}
| postfix_expression '.' nabla_item '(' Z_CONSTANT ')'{Y6($$,$1,$2,$3,$4,$5,$6)}
| postfix_expression '.' nabla_system {Y3($$,$1,$2,$3)}
//| postfix_expression PTR_OP IDENTIFIER {Y3($$,$1,$2,$3)}
| postfix_expression PTR_OP primary_expression {Y3($$,$1,$2,$3)} 
| postfix_expression INC_OP {Y2($$,$1,$2)}
| postfix_expression DEC_OP {Y2($$,$1,$2)}
//| mathlinks
| aleph_expression
;
///////////////////////////////////
// Unaries (operator,expression) //
///////////////////////////////////
unary_operator
: //N_ARY_CIRCLED_TIMES_OP | CENTER_DOT_OP
//|CIRCLED_ASTERISK_OP | CIRCLED_TIMES_OP
//| CROSS_OP | CROSS_OP_2D
 '⋅' | '*' | '+' | '-' | '~' | '!';
unary_expression
: postfix_expression {Y1($$,$1)}
//!| unary_expression SUPERSCRIPT_DIGIT_TWO {Ypow($$,$1,2)}
//!| unary_expression SUPERSCRIPT_DIGIT_THREE {Ypow($$,$1,3)}
| SQUARE_ROOT_OP unary_expression {Y2($$,$1,$2)}
| CUBE_ROOT_OP unary_expression {Y2($$,$1,$2)}
| INC_OP unary_expression {Y2($$,$1,$2)}
| DEC_OP unary_expression {Y2($$,$1,$2)}
// Permet d'insérer pour l'instant l'adrs() 
| '&' unary_expression {Yp2p($$,$1,$2)}
| unary_operator cast_expression {Y2($$,$1,$2)}
| SIZEOF unary_expression {Y2($$,$1,$2)}
| SIZEOF '(' type_name ')'{Y4($$,$1,$2,$3,$4)}
;
cast_expression
: unary_expression {Y1($$,$1)}
| '(' type_name ')' cast_expression {Y4($$,$1,$2,$3,$4)}
;
multiplicative_expression
: cast_expression {Y1($$,$1)}
| multiplicative_expression '*' cast_expression {Yop3p($$,$1,$2,$3)}
| multiplicative_expression '/' cast_expression {Yop3p($$,$1,$2,$3)}
| multiplicative_expression '%' cast_expression {Yop3p($$,$1,$2,$3)}
| multiplicative_expression CROSS_OP cast_expression {Yop3p($$,$1,$2,$3)}
| multiplicative_expression CROSS_OP_2D cast_expression {Yop3p($$,$1,$2,$3)}
| multiplicative_expression CENTER_DOT_OP cast_expression {Yop3p($$,$1,$2,$3)}
| multiplicative_expression CIRCLED_TIMES_OP cast_expression {Yop3p($$,$1,$2,$3)}
| multiplicative_expression CIRCLED_ASTERISK_OP cast_expression {Yop3p($$,$1,$2,$3)}
| multiplicative_expression N_ARY_CIRCLED_TIMES_OP cast_expression {Yop3p($$,$1,$2,$3)}
;
additive_expression
: multiplicative_expression {Y1($$,$1)}
|	additive_expression '+' multiplicative_expression{Yop3p($$,$1,$2,$3)}
|	additive_expression '-' multiplicative_expression{Yop3p($$,$1,$2,$3)}
;
shift_expression
: additive_expression {Y1($$,$1)}
|	shift_expression LSH_OP additive_expression{Y3($$,$1,$2,$3)}
|	shift_expression RSH_OP additive_expression{Y3($$,$1,$2,$3)}
;
relational_expression
: shift_expression {Y1($$,$1)}
|	relational_expression '<' shift_expression{Y3($$,$1,$2,$3)}
|	relational_expression '>' shift_expression{Y3($$,$1,$2,$3)}
|	relational_expression LEQ_OP shift_expression{Y3($$,$1,$2,$3)}
|	relational_expression GEQ_OP shift_expression{Y3($$,$1,$2,$3)}
;
equality_expression
: relational_expression {Y1($$,$1)}
| equality_expression EEQ_OP relational_expression{Y3($$,$1,$2,$3)}
| equality_expression NEQ_OP relational_expression{Y3($$,$1,$2,$3)}
;
and_expression
: equality_expression {Y1($$,$1)}
| and_expression '&' equality_expression{Y3($$,$1,$2,$3)}
;
exclusive_or_expression
: and_expression {Y1($$,$1)}
| exclusive_or_expression '^' and_expression{Y3($$,$1,$2,$3)}
;
inclusive_or_expression
: exclusive_or_expression {Y1($$,$1)}
| inclusive_or_expression '|' exclusive_or_expression {Y3($$,$1,$2,$3)}
;
logical_and_expression
: inclusive_or_expression {Y1($$,$1)}
| logical_and_expression AND_OP inclusive_or_expression {Y3($$,$1,$2,$3)}
;
logical_or_expression
: logical_and_expression {Y1($$,$1)}
| logical_or_expression IOR_OP logical_and_expression {Y3($$,$1,$2,$3)}
;
conditional_expression
: logical_or_expression {Y1($$,$1)}
| logical_or_expression '?' expression ':' conditional_expression {YopTernary5p($$,$1,$2,$3,$4,$5)}
;
///////////////////////////////////////
// Assignments (operator,expression) //
///////////////////////////////////////
assignment_operator
:  '=' {Y1($$,$1)}
| MUL_ASSIGN {Y1($$,$1)} | DIV_ASSIGN {Y1($$,$1)} | MOD_ASSIGN {Y1($$,$1)}
| ADD_ASSIGN {Y1($$,$1)} | SUB_ASSIGN {Y1($$,$1)}
| LSH_ASSIGN {Y1($$,$1)} | RSH_ASSIGN {Y1($$,$1)}
| AND_ASSIGN {Y1($$,$1)} | XOR_ASSIGN {Y1($$,$1)} | IOR_ASSIGN {Y1($$,$1)}
;
assignment_expression
: conditional_expression {Y1($$,$1)}
| unary_expression assignment_operator assignment_expression {Y3($$,$1,$2,$3)}
| unary_expression assignment_operator logical_or_expression '?' expression {YopDuaryExpression($$,$1,$2,$3,$5)}
| unary_expression assignment_operator type_specifier '(' initializer_list ')' {Y6($$,$1,$2,$3,$4,$5,$6)}
| unary_expression assignment_operator type_specifier '(' ')' {Y5($$,$1,$2,$3,$4,$5)}
;
expression
: assignment_expression {Y1($$,$1)}
| expression ',' assignment_expression {Y3($$,$1,$2,$3)}
;
constant_expression
: conditional_expression {Y1($$,$1)}
;


////////////////
// Statements //
////////////////
compound_statement
: '{' '}'{Y2($$,$1,$2)}
| start_scope statement_list end_scope {Y3($$,$1,$2,$3)}
| start_scope declaration_list end_scope {Y3($$,$1,$2,$3)}
| start_scope declaration_list statement_list end_scope{Y4($$,$1,$2,$3,$4)}
// Permet de rajouter des statements à la C++ avant la déclaration des variables locales
| start_scope statement declaration_list statement_list end_scope{Y5($$,$1,$2,$3,$4,$5)}
;
expression_statement
: ';'{Y1($$,$1)}
| expression ';'{Y2($$,$1,$2)}
| expression AT at_constant';' {Y4($$,$1,$2,$3,$4)}
;
selection_statement
: IF '(' expression ')' statement %prec REMOVE_SHIFT_REDUCE_MESSAGE_OF_IF_ELSE_AMBIGUITY {Y5($$,$1,$2,$3,$4,$5)}
| IF '(' expression ')' statement ELSE statement {Y7($$,$1,$2,$3,$4,$5,$6,$7)}
;
iteration_statement
: FOREACH nabla_item statement {Y3_foreach($$,$1,$2,$3)}
| FOREACH nabla_item AT at_constant statement {Y5_foreach($$,$1,$2,$3,$4,$5)}
| FOREACH nabla_matenv statement {Y3_foreach($$,$1,$2,$3)}
| FOREACH nabla_matenv AT at_constant statement {Y5_foreach($$,$1,$2,$3,$4,$5)}
| FOREACH IDENTIFIER CELL statement {Y4_foreach_cell_cell($$,$1,$2,$3,$4)}
| FOREACH IDENTIFIER NODE statement {Y4_foreach_cell_node($$,$1,$2,$3,$4)}
| FOREACH IDENTIFIER FACE statement {Y4_foreach_cell_face($$,$1,$2,$3,$4)}
| FOREACH IDENTIFIER PARTICLE statement {Y4_foreach_cell_particle($$,$1,$2,$3,$4)}
| WHILE '(' expression ')' statement {Y5($$,$1,$2,$3,$4,$5)}
| DO statement WHILE '(' expression ')' ';' {Y7($$,$1,$2,$3,$4,$5,$6,$7)}
| FOR '(' expression_statement expression_statement ')' statement {Y6($$,$1,$2,$3,$4,$5,$6)}
| FOR '(' expression_statement expression_statement expression ')' statement {Y7($$,$1,$2,$3,$4,$5,$6,$7)}
| FOR '(' type_specifier expression_statement expression_statement ')' statement {Y7($$,$1,$2,$3,$4,$5,$6,$7)}
| FOR '(' type_specifier expression_statement expression_statement expression ')' statement {Y8($$,$1,$2,$3,$4,$5,$6,$7,$8)}
;
jump_statement
: CONTINUE ';'{Y2($$,$1,$2)}
| BREAK ';'{Y2($$,$1,$2)}
| RETURN ';'{Y2($$,$1,$2)}
| RETURN expression ';'{Y3($$,$1,$2,$3)}
;
statement
: compound_statement {Y1($$,$1)}
| expression_statement {Y1($$,$1)}
| selection_statement {Y1($$,$1)}
| iteration_statement {Y1($$,$1)}
| jump_statement {Y1($$,$1)}
;
statement_list
: statement {Y1($$,$1)}
| statement_list statement {Y2($$,$1,$2)}
;


/////////////////
// ∇ functions //
/////////////////
function_definition
: declaration_specifiers declarator declaration_list compound_statement {Y4($$,$1,$2,$3,$4)}
| declaration_specifiers declarator declaration_list AT at_constant compound_statement {Y6($$,$1,$2,$3,$4,$5,$6)}
| declaration_specifiers declarator compound_statement {Y3($$,$1,$2,$3)}
| declaration_specifiers declarator AT at_constant compound_statement {Y5($$,$1,$2,$3,$4,$5)}
| declarator declaration_list compound_statement {Y3($$,$1,$2,$3)}
| declarator declaration_list AT at_constant compound_statement {Y5($$,$1,$2,$3,$4,$5)}
| declarator compound_statement {Y2($$,$1,$2)}
| declarator AT at_constant compound_statement {Y4($$,$1,$2,$3,$4)}
;


/////////////////////////
// ∇ items definitions //
/////////////////////////
nabla_item_definition
: nabla_items '{' nabla_item_declaration_list '}' ';' {Y2($$,$1,$3)};
nabla_item_declaration_list
: nabla_item_declaration {Y1($$,$1)}
| nabla_item_declaration_list nabla_item_declaration {Y2($$,$1,$2)};
nabla_direct_declarator
: IDENTIFIER {Y1($$,$1)}
| IDENTIFIER '[' nabla_items ']'{Y2($$,$1,$3)};
| IDENTIFIER '[' primary_expression ']'{Y2($$,$1,$3)};
nabla_item_declaration: type_name nabla_direct_declarator ';' {Y2($$,$1,$2)};


//////////////////////////
// ∇ options definition //
//////////////////////////
nabla_options_definition
: OPTIONS '{' nabla_option_declaration_list '}' ';' {Y1($$,$3)};
nabla_option_declaration_list
: nabla_option_declaration {Y1($$,$1)}
| nabla_option_declaration_list nabla_option_declaration {Y2($$,$1,$2)};
nabla_option_declaration
: type_specifier direct_declarator ';' {Y2($$,$1,$2)}
| type_specifier direct_declarator '=' expression ';' {Y4($$,$1,$2,$3,$4)}  
;


////////////////////////////
// ∇ materials definition //
////////////////////////////
nabla_materials_definition: MATERIALS '{' identifier_list '}' ';' {Y2($$,$1,$3)};


///////////////////////////////
// ∇ environments definition //
///////////////////////////////
nabla_environment_declaration
: IDENTIFIER '{' identifier_list '}' ';' {Y2($$,$1,$3)};
nabla_environments_declaration_list
: nabla_environment_declaration {Y1($$,$1)}
| nabla_environments_declaration_list nabla_environment_declaration {Y2($$,$1,$2)}
;
nabla_environments_definition: ENVIRONMENTS '{' nabla_environments_declaration_list '}' ';' {Y2($$,$1,$3)};


///////////////////////
// ∇ '@' definitions //
///////////////////////
at_single_constant
: Z_CONSTANT {Y1($$,$1)}
| R_CONSTANT {Y1($$,$1)}
| '-' Z_CONSTANT {Y2($$,$1,$2)}
| '+' Z_CONSTANT {Y2($$,$1,$2)}
| '-' R_CONSTANT {Y2($$,$1,$2)}
| '+' R_CONSTANT {Y2($$,$1,$2)};
at_constant
: at_single_constant {Yp1p($$,$1)}
| at_constant ',' at_single_constant {Yp3p($$,$1,$2,$3)};


////////////////////////
// ∇ jobs definitions //
////////////////////////
nabla_job_definition
: nabla_family declaration_specifiers IDENTIFIER '(' parameter_type_list ')' compound_statement
  {Y5_compound_job($$,$1,$2,$3,$5,$7)}
| nabla_family declaration_specifiers IDENTIFIER '(' parameter_type_list ')' nabla_parameter_list compound_statement
  {Y6_compound_job($$,$1,$2,$3,$5,$7,$8)}
| nabla_family declaration_specifiers IDENTIFIER '(' parameter_type_list ')' AT at_constant compound_statement
  {Y7_compound_job($$,$1,$2,$3,$5,$7,$8,$9)}
| nabla_family declaration_specifiers IDENTIFIER '(' parameter_type_list ')' AT at_constant IF '(' constant_expression ')' compound_statement
  {Y11_compound_job($$,$1,$2,$3,$5,$7,$8,$9,$10,$11,$12,$13)}
| nabla_family declaration_specifiers IDENTIFIER '(' parameter_type_list ')' nabla_parameter_list AT at_constant compound_statement
  {Y8_compound_job($$,$1,$2,$3,$5,$7,$8,$9,$10)}
| nabla_family declaration_specifiers IDENTIFIER '(' parameter_type_list ')' nabla_parameter_list AT at_constant IF '(' constant_expression ')' compound_statement
  {Y12_compound_job($$,$1,$2,$3,$5,$7,$8,$9,$10,$11,$12,$13,$14)}
;


/////////////////
// ∇ libraries //
/////////////////
single_library:
| LIB_DFT         {Y1($$,$1)}
| LIB_GMP         {Y1($$,$1)}
| LIB_MPI         {Y1($$,$1)}
| MAIL            {Y1($$,$1)}
| PARTICLES       {Y1($$,$1)}
| LIB_ALEPH       {Y1($$,$1)}
| LIB_SLURM       {Y1($$,$1)}
| LIB_MATENV      {Y1($$,$1)}
| LIB_CARTESIAN   {Y1($$,$1)}
| LIB_MATHEMATICA {Y1($$,$1)}
;
with_library_list: single_library
| with_library_list ',' single_library{Y2($$,$1,$3)};
with_library: WITH with_library_list ';'{Y3($$,$1,$2,$3)};


///////////////
// ∇ grammar //
///////////////
nabla_grammar
: declaration				        {Y1($$,$1)}
| with_library                  {Y1($$,$1)}
| nabla_options_definition      {Y1($$,$1)}
| nabla_item_definition         {Y1($$,$1)}
| nabla_materials_definition    {Y1($$,$1)}
| nabla_environments_definition {Y1($$,$1)}
| function_definition	        {Y1($$,$1)}
| nabla_job_definition          {Y1($$,$1)}
;


///////////////////////
// Aleph Expressions //
///////////////////////
aleph_vector
: ALEPH_RHS {Y1($$,$1)}
| ALEPH_LHS {Y1($$,$1)}
;

aleph_expression
: aleph_vector {Y1($$,$1)} // Utilisé pour dumper par exemple
| LIB_ALEPH aleph_vector ALEPH_RESET {Y3($$,$1,$2,$3)}
| LIB_ALEPH ALEPH_SOLVE {Y2($$,$1,$2)}
| LIB_ALEPH aleph_vector ALEPH_NEW_VALUE {Y3($$,$1,$2,$3)}
| LIB_ALEPH aleph_vector ALEPH_ADD_VALUE {Y3($$,$1,$2,$3)}
| LIB_ALEPH aleph_vector ALEPH_SET {Y3($$,$1,$2,$3)}
| LIB_ALEPH ALEPH_MTX ALEPH_ADD_VALUE {Y3($$,$1,$2,$3)}
| LIB_ALEPH ALEPH_MTX ALEPH_SET {Y3($$,$1,$2,$3)}
| LIB_ALEPH ALEPH_LHS ALEPH_GET {Y3($$,$1,$2,$3)}
| LIB_ALEPH ALEPH_RHS ALEPH_GET {Y3($$,$1,$2,$3)}
;


///////////////////////////
// Junk to look at later //
///////////////////////////

  

/*
struct_declaration
: specifier_qualifier_list struct_declarator_list ';'{Y2($$,$1,$2)}
;
struct_declaration_list
:	struct_declaration {Y1($$,$1)}
|	struct_declaration_list struct_declaration{Y2($$,$1,$2)}
;

// ENUMERATORS
enumerator
: IDENTIFIER {Y1($$,$1)}
| IDENTIFIER '=' constant_expression{Y3($$,$1,$2,$3)}
;
enumerator_list
: enumerator {Y1($$,$1)}
| enumerator_list ',' enumerator{Y3($$,$1,$2,$3)}
;
enum_specifier
: ENUM '{' enumerator_list '}'{Y4($$,$1,$2,$3,$4)}
| ENUM IDENTIFIER '{' enumerator_list '}'{Y5($$,$1,$2,$3,$4,$5)}
| ENUM IDENTIFIER{Y2($$,$1,$2)}
;

// SPECIFIERS
struct_or_union
: STRUCT {Y1($$,$1)}
| UNION {Y1($$,$1)}
;

// Structs or Unions
struct_or_union_specifier
: struct_or_union IDENTIFIER '{' struct_declaration_list '}'{Y5($$,$1,$2,$3,$4,$5)}
| struct_or_union '{' struct_declaration_list '}'{Y4($$,$1,$2,$3,$4)}
| struct_or_union IDENTIFIER{Y2($$,$1,$2)}
;

mathlinks:
| MATHLINK PRIME '[' IDENTIFIER ']' {primeY1ident($$,$4)}
| MATHLINK PRIME '[' Z_CONSTANT ']' {primeY1($$,$4)}
;

*/

%%


/*****************************************************************************
 * tokenidToRuleid
 *****************************************************************************/
inline int tokenidToRuleid(int tokenid){
  return YYTRANSLATE(tokenid);
}


/*****************************************************************************
 * yyTranslate
 *****************************************************************************/
inline int yyTranslate(int tokenid){
  return YYTRANSLATE(tokenid);
}


/*****************************************************************************
 * yyUndefTok
 *****************************************************************************/
inline int yyUndefTok(void){
  return YYUNDEFTOK;
}


/*****************************************************************************
 * yyNameTranslate
 *****************************************************************************/
inline int yyNameTranslate(int tokenid){
  return yytname[YYTRANSLATE(tokenid)][1];
}


/*****************************************************************************
 * rulenameToId
 *****************************************************************************/
int rulenameToId(const char *rulename){
  register unsigned int i;
  register size_t rnLength=strlen(rulename);
  for(i=0; yytname[i]!=NULL;i+=1){
    if (strlen(yytname[i])!=rnLength) continue;
    if (strcmp(yytname[i], rulename)!=0) continue;
    return i;
  }
  dbg("[rulenameToId] error with '%s'",rulename);
  return 1; // error
}


/*****************************************************************************
 * tokenToId
 *****************************************************************************/
int tokenToId(const char *token){
  register unsigned int i;
  register size_t rnLength=strlen(token);
  for(i=0; yytname[i]!=NULL;++i){
    if (strlen(yytname[i])!=rnLength) continue;
    if (strcmp(yytname[i], token)!=0) continue;
    return i;
  }
  dbg("[tokenToId] error with '%s'",token);
  return 1; // error
}
