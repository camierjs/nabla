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
#include "nablaY.h" 
#include <stdio.h>
#include <string.h>

#undef YYDEBUG
#define YYSTYPE astNode*
int yylineno;
char nabla_input_file[1024];
 
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
%token NULL_ASSIGN MIN_ASSIGN MAX_ASSIGN
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
%token FILECALL FILETYPE

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
%right REAL REAL3 REAL3x3 '('

%%

/////////////////////////////////
// Entry point of input stream //
/////////////////////////////////
entry_point: nabla_inputstream{*root=$1;};
nabla_inputstream: nabla_grammar {rhs}
| nabla_inputstream nabla_grammar {astAddChild($1,$2);};


///////////////////////////
// ∇ scopes: std & std+@ //
///////////////////////////
start_scope: '{' {rhs};
end_scope: '}' {rhs} | '}' AT at_constant {rhs};


//////////////////////////////////////////////////
// ∇ types, qualifiers, specifier, lists & name //
//////////////////////////////////////////////////
type_specifier
: VOID {rhs}
| CHAR {rhs} 
| SHORT {rhs}
| INT {rhs}
| LONG {rhs}
| FLOAT {rhs}
| DOUBLE {rhs}
| SIGNED {rhs}
| UNSIGNED {rhs}
| BOOL {rhs}
| SIZE_T {rhs}
| REAL { if (type_precise) preciseY1($$,GMP_REAL) else rhs; type_precise=type_volatile=false;}
| INTEGER {
    if (type_precise){
      if (type_volatile) volatilePreciseY1($$,GMP_INTEGER)
      else preciseY1($$,GMP_INTEGER)
    }else rhs;
    type_precise=type_volatile=false;
  }
| INT32 {rhs}
| INT64 {rhs}
| REAL2 {rhs}
| REAL3 {rhs}
| REAL3x3 {rhs}
| REAL2x2 {rhs}
| CELLTYPE {rhs}
| NODETYPE {rhs}
| PARTICLETYPE {rhs}
| FACETYPE {rhs}
| UIDTYPE {rhs}
| FILETYPE {rhs} 
| FILECALL '(' IDENTIFIER ',' IDENTIFIER ')' {rhs} 
| MATERIAL {rhs}
//| struct_or_union_specifier {rhs}
//| enum_specifier {rhs}
;

storage_class_specifier
: EXTERN {rhs}
| STATIC {rhs}
| AUTO {rhs}
| INLINE {rhs}
| REGISTER {rhs}
;
type_qualifier
: CONST {rhs}
| ALIGNED {rhs}
| VOLATILE {rhs;type_volatile=true;}
| GMP_PRECISE {rhs;type_precise=true;}
;
type_qualifier_list:
  type_qualifier {rhs}
| type_qualifier_list type_qualifier {rhs}
;
specifier_qualifier_list
: type_specifier {rhs}
| type_specifier specifier_qualifier_list{rhs}
| type_qualifier {rhs}
| type_qualifier specifier_qualifier_list{rhs}
;
type_name
: specifier_qualifier_list {rhs}
| specifier_qualifier_list abstract_declarator{rhs}
;


///////////////////////////////////////////////////////////
// ∇ item(s), group, region, family & system definitions //
///////////////////////////////////////////////////////////
nabla_matenv: MATERIAL {rhs}| ENVIRONMENT {rhs};
nabla_matenvs: MATERIALS {rhs} | ENVIRONMENTS {rhs};
nabla_item
: CELL {rhs}
| NODE {rhs}
| FACE {rhs}
| PARTICLE {rhs}
;
nabla_items
: CELLS {rhs}
| NODES {rhs}
| FACES {rhs}
| GLOBAL {rhs}
| PARTICLES {rhs}
;
nabla_scope: OWN {rhs} | ALL {rhs};
nabla_region: INNER {rhs} | OUTER {rhs};
nabla_family
: nabla_items {rhs}
| nabla_matenvs {rhs}
| nabla_scope nabla_items {rhs}
| nabla_region nabla_items {rhs}
| nabla_scope nabla_region nabla_items {rhs}
;
nabla_system
: LID {rhs}
| SID {rhs}
| UID {rhs}
| THIS{rhs}
| NBNODE{rhs}
| NBCELL{rhs}
//| FATAL{rhs}
| BOUNDARY_CELL{rhs}
| BACKCELL{rhs}
| BACKCELLUID{rhs}
| FRONTCELL{rhs}
| FRONTCELLUID{rhs}
| NEXTCELL {rhs}
| PREVCELL {rhs}
| NEXTNODE {rhs}
| PREVNODE {rhs}
| PREVLEFT {rhs}
| PREVRIGHT {rhs}
| NEXTLEFT {rhs}
| NEXTRIGHT {rhs}
| TIME {rhs}
| TIME REMAIN {remainY1($$)}
| TIME LIMIT {limitY1($$)}
| EXIT {rhs}
| ITERATION {rhs}
| MAIL {rhs}
;


//////////////
// Pointers //
//////////////
pointer
: '*' {rhs}
| '*' type_qualifier_list{rhs}
| '*' RESTRICT {rhs}
| '*' type_qualifier_list pointer{rhs}
;

//////////////////
// INITIALIZERS //
//////////////////
initializer
: assignment_expression {rhs}
| '{' initializer_list '}'{rhs}
//| type_specifier '(' initializer_list ')'{Y4($$,$1,$2,$3,$4)}
;
initializer_list
: initializer {rhs}
| initializer_list ',' initializer {rhs}
;

//////////////////
// PREPROCESSOR //
//////////////////
preproc
: PREPROCS {
  int n;
  rhs;
  //printf("%s",$1->token);
  if ((n=sscanf($1->token, "# %d \"%[^\"]\"", &yylineno, nabla_input_file))!=2)
    error(!0,0,"declaration sscanf error!");
  //printf("%s:%d:\n",nabla_input_file,yylineno);
  }
;

//////////////////
// DeclaraTIONS //
//////////////////
declaration_specifiers
: storage_class_specifier {rhs}
| storage_class_specifier declaration_specifiers{rhs}
| type_specifier {rhs}
| type_specifier declaration_specifiers{rhs}
| type_qualifier {rhs}
| type_qualifier declaration_specifiers{rhs}
;
declaration
// On patche l'espace qui nous a été laissé par le sed pour remettre le bon #include
: INCLUDES {$1->token[0]='#';rhs}
| preproc {rhs}
| declaration_specifiers ';' {rhs}
| declaration_specifiers init_declarator_list ';' {rhs}  
;
declaration_list
: declaration{rhs}
| declaration_list declaration{rhs}
;


/////////////////
// DeclaraTORS //
/////////////////
declarator
: pointer direct_declarator{rhs}
| direct_declarator {rhs}
;
// Identifier list for direct declarators
identifier_list
: IDENTIFIER {rhs}
| identifier_list ',' IDENTIFIER	{rhs}
;
direct_declarator
: IDENTIFIER {rhs}
| '(' declarator ')'{rhs}
| direct_declarator '[' constant_expression ']'{Y4($$,$1,$2,$3,$4)}
| direct_declarator '[' ']'{rhs}
| direct_declarator '(' parameter_type_list ')'{Y4($$,$1,$2,$3,$4)}
| direct_declarator '(' identifier_list ')'{Y4($$,$1,$2,$3,$4)}
| direct_declarator '(' ')'{rhs}
;
init_declarator
:	declarator {rhs}
// Permet de faire des appels constructeurs à-là '= Real3(0.0,0.0,0.0)' lors des déclarations
//|	declarator '=' type_specifier assignment_expression{Y4($$,$1,$2,$3,$4)} // initializer
// Permet de faire des appels constructeurs à-là '= Real3x3()' lors des déclarations
//|	declarator '=' type_specifier '(' ')' {Y5($$,$1,$2,$3,$4,$5)}
|	declarator '=' initializer {rhs}
;
init_declarator_list
:	init_declarator {rhs}
|	init_declarator_list ',' init_declarator{rhs}
;
abstract_declarator
: 	pointer {rhs}
|	direct_abstract_declarator {rhs}
|	pointer direct_abstract_declarator{rhs}
;
direct_abstract_declarator
: '(' abstract_declarator ')'{rhs}
| '[' ']'{rhs}
| '[' constant_expression ']'{rhs}
| direct_abstract_declarator '[' ']'{rhs}
| direct_abstract_declarator '[' constant_expression ']'{Y4($$,$1,$2,$3,$4)}
| '(' ')'{rhs}
| '(' parameter_type_list ')'{rhs}
| direct_abstract_declarator '(' ')'{rhs}
| direct_abstract_declarator '(' parameter_type_list ')'{Y4($$,$1,$2,$3,$4)}
;


////////////////////
// Std parameters //
////////////////////
parameter_type_list
:	parameter_list {rhs}
|	parameter_list ',' ELLIPSIS{rhs}
;
parameter_list
:	parameter_declaration {rhs}
|	parameter_list ',' parameter_declaration{rhs}
;
parameter_declaration
: nabla_xyz_declaration {rhs}
| nabla_mat_declaration {rhs}
| nabla_env_declaration {rhs}
| declaration_specifiers declarator {rhs}
| declaration_specifiers abstract_declarator {rhs}
| declaration_specifiers {rhs}
;


//////////////////////////////
// ∇ xyz/mat/env parameters //
//////////////////////////////
nabla_xyz_direction:IDENTIFIER {rhs};
nabla_xyz_declaration: XYZ nabla_xyz_direction {rhs};
nabla_mat_material:IDENTIFIER {rhs};
nabla_mat_declaration: MAT nabla_mat_material {rhs};
nabla_env_environment:IDENTIFIER {rhs};
nabla_env_declaration: ENV nabla_env_environment {rhs};
nabla_parameter_declaration
: nabla_item direct_declarator {rhs};
nabla_parameter_list
: nabla_parameter_declaration {rhs}
| nabla_parameter_list ',' nabla_parameter_declaration {Y2($$,$1,$3)};

// !? WTF between these two 'nabla_parameter_list' ?!

/////////////////////////
// ∇ IN/OUT parameters //
/////////////////////////
nabla_parameter
: nabla_in_parameter_list {rhs}
| nabla_out_parameter_list {rhs}
| nabla_inout_parameter_list {rhs};
nabla_parameter_list
: nabla_parameter {rhs}
| nabla_parameter_list nabla_parameter{rhs};
nabla_in_parameter_list: IN '(' nabla_parameter_list ')' {Y2($$,$1,$3)}; 
nabla_out_parameter_list: OUT '(' nabla_parameter_list ')' {Y2($$,$1,$3)};  
nabla_inout_parameter_list: INOUT '(' nabla_parameter_list ')' {Y2($$,$1,$3)};  


////////////////
// NAMESPACES //
////////////////
//| NAMESPACE IDENTIFIER {rhs}
//| IDENTIFIER '<' REAL '>' NAMESPACE IDENTIFIER {Y1($$,$6)}
//| IDENTIFIER NAMESPACE IDENTIFIER {rhs}

argument_expression_list
: assignment_expression {rhs}
| argument_expression_list ',' assignment_expression {rhs}
;

/////////////////
// EXPRESSIONS //
/////////////////
primary_expression
: BUILTIN_INFF {rhs}
| FRACTION_ONE_HALF_CST {rhs}
| FRACTION_ONE_THIRD_CST {rhs}
| FRACTION_ONE_QUARTER_CST {rhs}
| FRACTION_ONE_EIGHTH_CST {rhs}
| DIESE {rhs}  // Permet d'écrire un '#' à la place d'un [c|n]
| IDENTIFIER {rhs}
| nabla_item {rhs} // Permet de rajouter les items Nabla au sein des corps de fonctions
| nabla_system {rhs}
| HEX_CONSTANT {rhs} 
| OCT_CONSTANT {rhs}
| Z_CONSTANT {rhs}
| R_CONSTANT {rhs}
| QUOTE_LITERAL {rhs}
| STRING_LITERAL {rhs}
| '(' expression ')'	{rhs}
;
postfix_expression
: primary_expression {rhs}
| postfix_expression FOREACH_NODE_INDEX {rhs}
| postfix_expression FOREACH_CELL_INDEX {rhs}
| postfix_expression '[' expression ']' {Y4($$,$1,$2,$3,$4)}
| REAL '(' ')'{rhs}
| REAL '(' expression ')' {Y4($$,$1,$2,$3,$4)}
| REAL3 '(' ')'{rhs}
| REAL3 '(' expression ')' {Y4($$,$1,$2,$3,$4)}
| REAL3x3 '(' ')'{rhs}
| REAL3x3 '(' expression ')' {Y4($$,$1,$2,$3,$4)}
| postfix_expression '(' ')' {rhs}
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
| postfix_expression '.' IDENTIFIER {rhs}
| postfix_expression '.' nabla_item '(' Z_CONSTANT ')'{rhs}
| postfix_expression '.' nabla_system {rhs}
| postfix_expression PTR_OP primary_expression {rhs} 
| postfix_expression INC_OP {rhs}
| postfix_expression DEC_OP {rhs}
| postfix_expression SUPERSCRIPT_DIGIT_TWO {Ypow($$,$1,2)}
| postfix_expression SUPERSCRIPT_DIGIT_THREE {Ypow($$,$1,3)}
//| mathlinks
| aleph_expression
;
///////////////////////////////////
// Unaries (operator,expression) //
///////////////////////////////////
unary_prefix_operator: '⋅' | '*' | '+' | '-' | '~' | '!';
unary_expression
: postfix_expression {rhs}
| SQUARE_ROOT_OP unary_expression {rhs}
| CUBE_ROOT_OP unary_expression {rhs}
| INC_OP unary_expression {rhs}
| DEC_OP unary_expression {rhs}
// Permet d'insérer pour l'instant l'adrs() 
| '&' unary_expression {Yp2p($$,$1,$2)}
| unary_prefix_operator cast_expression {rhs}
| SIZEOF unary_expression {rhs}
| SIZEOF '(' type_name ')'{Y4($$,$1,$2,$3,$4)}
;
cast_expression
: unary_expression {rhs}
| '(' type_name ')' cast_expression {Y4($$,$1,$2,$3,$4)}
;
multiplicative_expression
: cast_expression {rhs}
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
: multiplicative_expression {rhs}
| additive_expression '+' multiplicative_expression {Yop3p($$,$1,$2,$3)}
| additive_expression '-' multiplicative_expression {Yop3p($$,$1,$2,$3)}
;
shift_expression
: additive_expression {rhs}
| shift_expression LSH_OP additive_expression{rhs}
| shift_expression RSH_OP additive_expression{rhs}
;
relational_expression
: shift_expression {rhs}
| relational_expression '<' shift_expression{rhs}
| relational_expression '>' shift_expression{rhs}
| relational_expression LEQ_OP shift_expression{rhs}
| relational_expression GEQ_OP shift_expression{rhs}
;
equality_expression
: relational_expression {rhs}
| equality_expression EEQ_OP relational_expression{rhs}
| equality_expression NEQ_OP relational_expression{rhs}
;
and_expression
: equality_expression {rhs}
| and_expression '&' equality_expression{rhs}
;
exclusive_or_expression
: and_expression {rhs}
| exclusive_or_expression '^' and_expression{rhs}
;
inclusive_or_expression
: exclusive_or_expression {rhs}
| inclusive_or_expression '|' exclusive_or_expression {rhs}
;
logical_and_expression
: inclusive_or_expression {rhs}
| logical_and_expression AND_OP inclusive_or_expression {rhs}
;
logical_or_expression
: logical_and_expression {rhs}
| logical_or_expression IOR_OP logical_and_expression {rhs}
;
conditional_expression
: logical_or_expression {rhs}
| logical_or_expression '?' expression ':' conditional_expression {YopTernary5p($$,$1,$2,$3,$4,$5)}
;
///////////////////////////////////////
// Assignments (operator,expression) //
///////////////////////////////////////
assignment_operator
:  '=' {rhs}
| MUL_ASSIGN {rhs} | DIV_ASSIGN {rhs} | MOD_ASSIGN {rhs}
| ADD_ASSIGN {rhs} | SUB_ASSIGN {rhs}
| LSH_ASSIGN {rhs} | RSH_ASSIGN {rhs}
| AND_ASSIGN {rhs} | XOR_ASSIGN {rhs} | IOR_ASSIGN {rhs}
| NULL_ASSIGN {rhs}
| MIN_ASSIGN {rhs}
//| MAX_ASSIGN {Z1(MAX_ASSIGN, $$,$1)}
;
assignment_expression
: conditional_expression {rhs}
| unary_expression assignment_operator assignment_expression {rhs}
| unary_expression assignment_operator logical_or_expression '?' expression {YopDuaryExpression($$,$1,$2,$3,$5)}
// Permet de faire des appels constructeurs à-là 'Real3(0.0,0.0,0.0)' lors des expressions
//| unary_expression assignment_operator type_specifier '(' initializer_list ')' {rhs}
//| unary_expression assignment_operator type_specifier '(' ')' {Y5($$,$1,$2,$3,$4,$5)}
;

expression
: assignment_expression {rhs}
| expression ',' assignment_expression {rhs}
;
constant_expression
: conditional_expression {rhs}
;


////////////////
// Statements //
////////////////
compound_statement
: start_scope end_scope {rhs}
| start_scope statement_list end_scope {rhs}
| start_scope declaration_list end_scope {rhs}
| start_scope declaration_list statement_list end_scope{Y4($$,$1,$2,$3,$4)}
// Permet de rajouter des statements à la C++ avant la déclaration des variables locales
| start_scope statement_list declaration_list statement_list end_scope{Y5($$,$1,$2,$3,$4,$5)}
;
expression_statement
: ';'{rhs}
| expression ';'{rhs}
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
| FOR '(' expression_statement expression_statement ')' statement {rhs}
| FOR '(' expression_statement expression_statement expression ')' statement {Y7($$,$1,$2,$3,$4,$5,$6,$7)}
| FOR '(' type_specifier expression_statement expression_statement ')' statement {Y7($$,$1,$2,$3,$4,$5,$6,$7)}
| FOR '(' type_specifier expression_statement expression_statement expression ')' statement {Y8($$,$1,$2,$3,$4,$5,$6,$7,$8)}
;
jump_statement
: CONTINUE ';'{rhs}
| BREAK ';'{rhs}
| RETURN ';'{rhs}
| RETURN expression ';'{rhs}
//| RETURN type_specifier expression ';' {Y4($$,$1,$2,$3,$4)}
;
statement
: compound_statement {rhs}
| expression_statement {rhs}
| selection_statement {rhs}
| iteration_statement {rhs}
| jump_statement {rhs}
;
statement_list
: statement {rhs}
| statement_list statement {rhs}
;


/////////////////
// ∇ functions //
/////////////////
function_definition
: declaration_specifiers declarator declaration_list compound_statement {Y4($$,$1,$2,$3,$4)}
| declaration_specifiers declarator declaration_list AT at_constant compound_statement {rhs}
| declaration_specifiers declarator compound_statement {rhs}
| declaration_specifiers declarator AT at_constant compound_statement {Y5($$,$1,$2,$3,$4,$5)}
//| declarator declaration_list compound_statement {rhs}
//| declarator declaration_list AT at_constant compound_statement {Y5($$,$1,$2,$3,$4,$5)}
//| declarator compound_statement {rhs}
//| declarator AT at_constant compound_statement {Y4($$,$1,$2,$3,$4)}
;


/////////////////////////
// ∇ items definitions //
/////////////////////////
nabla_item_definition
: nabla_items '{' nabla_item_declaration_list '}' ';' {Y2($$,$1,$3)};
nabla_item_declaration_list
: nabla_item_declaration {rhs}
| nabla_item_declaration_list nabla_item_declaration {rhs}
;
nabla_direct_declarator
: IDENTIFIER {rhs}
| IDENTIFIER '[' nabla_items ']'{Y2($$,$1,$3)}
| IDENTIFIER '[' primary_expression ']'{Y2($$,$1,$3)}
;
nabla_item_declaration
: type_name nabla_direct_declarator ';' {rhs}
| preproc {rhs}
;


//////////////////////////
// ∇ options definition //
//////////////////////////
nabla_options_definition
: OPTIONS '{' nabla_option_declaration_list '}' ';' {Y1($$,$3)};
nabla_option_declaration_list
: nabla_option_declaration {rhs}
| nabla_option_declaration_list nabla_option_declaration {rhs};
nabla_option_declaration
: type_specifier direct_declarator ';' {rhs}
| type_specifier direct_declarator '=' expression ';' {Y4($$,$1,$2,$3,$4)}  
| preproc {rhs}
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
: nabla_environment_declaration {rhs}
| nabla_environments_declaration_list nabla_environment_declaration {rhs}
;
nabla_environments_definition: ENVIRONMENTS '{' nabla_environments_declaration_list '}' ';' {Y2($$,$1,$3)};


///////////////////////
// ∇ '@' definitions //
///////////////////////
at_single_constant
: Z_CONSTANT {rhs}
| R_CONSTANT {rhs}
| '-' Z_CONSTANT {rhs}
| '+' Z_CONSTANT {rhs}
| '-' R_CONSTANT {rhs}
| '+' R_CONSTANT {rhs};
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
| LIB_DFT         {rhs}
| LIB_GMP         {rhs}
| LIB_MPI         {rhs}
| MAIL            {rhs}
| PARTICLES       {rhs}
| LIB_ALEPH       {rhs}
| LIB_SLURM       {rhs}
| LIB_MATENV      {rhs}
| LIB_CARTESIAN   {rhs}
| LIB_MATHEMATICA {rhs}
;
with_library_list: single_library
| with_library_list ',' single_library{Y2($$,$1,$3)};
with_library: WITH with_library_list ';'{rhs};


///////////////
// ∇ grammar //
///////////////
nabla_grammar
: with_library                  {rhs}
| declaration                   {rhs}
| nabla_options_definition      {rhs}
| nabla_item_definition         {rhs}
| nabla_materials_definition    {rhs}
| nabla_environments_definition {rhs}
| function_definition	        {rhs}
| nabla_job_definition          {rhs}
;


///////////////////////
// Aleph Expressions //
///////////////////////
aleph_vector
: ALEPH_RHS {rhs}
| ALEPH_LHS {rhs}
;

aleph_expression
: aleph_vector {rhs} // Utilisé pour dumper par exemple
| LIB_ALEPH aleph_vector ALEPH_RESET {rhs}
| LIB_ALEPH ALEPH_SOLVE {rhs}
| LIB_ALEPH aleph_vector ALEPH_NEW_VALUE {rhs}
| LIB_ALEPH aleph_vector ALEPH_ADD_VALUE {rhs}
| LIB_ALEPH aleph_vector ALEPH_SET {rhs}
| LIB_ALEPH ALEPH_MTX ALEPH_ADD_VALUE {rhs}
| LIB_ALEPH ALEPH_MTX ALEPH_SET {rhs}
| LIB_ALEPH ALEPH_LHS ALEPH_GET {rhs}
| LIB_ALEPH ALEPH_RHS ALEPH_GET {rhs}
;


/////////////////////////////
// STRUCTS, ENUMS & UNIONS //
/////////////////////////////

/*
struct_declaration
: specifier_qualifier_list struct_declarator_list ';'{rhs}
;
struct_declaration_list
:	struct_declaration {rhs}
|	struct_declaration_list struct_declaration{rhs}
;

// ENUMERATORS
enumerator
: IDENTIFIER {rhs}
| IDENTIFIER '=' constant_expression{rhs}
;
enumerator_list
: enumerator {rhs}
| enumerator_list ',' enumerator{rhs}
;
enum_specifier
: ENUM '{' enumerator_list '}'{Y4($$,$1,$2,$3,$4)}
| ENUM IDENTIFIER '{' enumerator_list '}'{Y5($$,$1,$2,$3,$4,$5)}
| ENUM IDENTIFIER{rhs}
;

// SPECIFIERS
struct_or_union
: STRUCT {rhs}
| UNION {rhs}
;

// Structs or Unions
struct_or_union_specifier
: struct_or_union IDENTIFIER '{' struct_declaration_list '}'{Y5($$,$1,$2,$3,$4,$5)}
| struct_or_union '{' struct_declaration_list '}'{Y4($$,$1,$2,$3,$4)}
| struct_or_union IDENTIFIER{rhs}
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


/*****************************************************************************
 * tokenToId
 *****************************************************************************/
int tokenToId(const char *token){
  unsigned int i;
  const size_t rnLength=strlen(token);
  for(i=0; yytname[i]!=NULL;++i){
    if (strlen(yytname[i])!=rnLength) continue;
    if (strcmp(yytname[i], token)!=0) continue;
    return i;
  }
  dbg("[tokenToId] error with '%s'",token);
  return 1; // error
}



// *****************************************************************************
// * Z_CALL
// * YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX
// * YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
// * YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to token YYLEX-NUM
// * YYR1[YYN] -- Symbol number of symbol that rule YYN derives
// * YYR2[YYN] -- Number of symbols composing right hand side of rule YYN
// * YYSTYPE lhs=yyval;
// *****************************************************************************
inline void rhsLinear(int yyn, astNode **yyval, astNode* *yyvsp){
  int yyi;
  // Nombre d'éléments dans notre RHS
  const int yynrhs = yyr2[yyn];
  // On accroche le nouveau noeud
  astNode *first=*yyval=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);
  // On va scruter tous les éléments
  // On commence par rajouter le premier comme fils
  astNode *next=yyvsp[(0+1)-(yynrhs)];
  astAddChild(first,next);
  first=next;
  // Et les autres comme frères
  for(yyi=1; yyi<yynrhs; yyi++){
    next=yyvsp[(yyi+1)-(yynrhs)];
    astAddNext(first,next);
    first=next;
  }
}
