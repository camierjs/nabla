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
bool type_volatile=false;
bool type_precise=false;
bool adrs_it=false;
 
bool typedef_hit=false;
nablaType *typedef_names=NULL;
extern char *last_identifier;
%}
 
///////////////////////////////
// Terminals used in grammar //
///////////////////////////////

// C-GRAMMAR
%token SPACE PREPROCS INCLUDES 
%token IDENTIFIER STRING_LITERAL QUOTE_LITERAL SIZEOF
%token PTR_OP INC_OP DEC_OP LSH_OP RSH_OP LEQ_OP GEQ_OP EEQ_OP NEQ_OP
%token AND_OP IOR_OP MUL_ASSIGN DIV_ASSIGN MOD_ASSIGN ADD_ASSIGN
%token NULL_ASSIGN MIN_ASSIGN MAX_ASSIGN
%token SUB_ASSIGN LSH_ASSIGN RSH_ASSIGN AND_ASSIGN
%token XOR_ASSIGN IOR_ASSIGN
%token TYPEDEF EXTERN STATIC AUTO REGISTER INLINE
%token CHAR SHORT INT LONG SIGNED UNSIGNED FLOAT DOUBLE CONST VOLATILE VOID
%token CASE DEFAULT IF ELSE SWITCH WHILE DO FOR GOTO CONTINUE BREAK RETURN
%token HEX_CONSTANT OCT_CONSTANT Z_CONSTANT R_CONSTANT
%token CALL END_OF_CALL ADRS_IN ADRS_OUT POSTFIX_CONSTANT POSTFIX_CONSTANT_VALUE
%token PREFIX_PRIMARY_CONSTANT POSTFIX_PRIMARY_CONSTANT
%token STRUCT UNION ENUM ELLIPSIS
%token TYPEDEF_NAME
%token GLOBAL RESTRICT ALIGNED
%token ATTRIBUTE ASM // GNU_VA_LIST
%token IS IS_OP_INI IS_OP_END
%token SET_INI SET_END

 // MATHS tokens
%token SQUARE_ROOT_OP CUBE_ROOT_OP N_ARY_CIRCLED_TIMES_OP
%token CENTER_DOT_OP CROSS_OP CROSS_OP_2D CIRCLED_TIMES_OP CIRCLED_ASTERISK_OP
%token FRACTION_ONE_HALF_CST FRACTION_ONE_THIRD_CST
%token FRACTION_ONE_QUARTER_CST FRACTION_ONE_EIGHTH_CST
%token BUILTIN_INFF
%token SUPERSCRIPT_DIGIT_TWO SUPERSCRIPT_DIGIT_THREE

 // SPECIFIC NABLA GRAMMAR
%token COMPOUND_JOB_INI COMPOUND_REDUCTION_INI
%token COMPOUND_JOB_END COMPOUND_REDUCTION_END
%token COORDS OPTIONS
%token AT DIESE
%token IN OUT INOUT
%token ALL OWN INNER OUTER
%token BOOL NATURAL INTEGER INT32 INT64 REAL REAL2 REAL2x2 REAL3 REAL3x3 UIDTYPE SIZE_T
%token CELLTYPE NODETYPE FACETYPE
%token CELL CELLS FACE FACES NODE NODES
%token FORALL FORALL_INI FORALL_END FORALL_NODE_INDEX FORALL_CELL_INDEX FORALL_MTRL_INDEX
%token PARTICLE PARTICLES PARTICLETYPE
%token FILECALL FILETYPE OFSTREAM

 // Nabla SUPERSCRIPT_*
%token SUPERSCRIPT_N_PLUS_ONE

 // Nabla Cartesian
%token XYZ NEXTNODE PREVNODE PREVLEFT PREVRIGHT NEXTLEFT NEXTRIGHT
%token NEXTCELL PREVCELL
%token NEXTCELL_X PREVCELL_X NEXTCELL_Y PREVCELL_Y NEXTCELL_Z PREVCELL_Z

 // Nabla LIBRARIES
%token LIB_MPI LIB_ALEPH LIB_CARTESIAN LIB_GMP LIB_MATHEMATICA LIB_SLURM MAIL LIB_MAIL LIB_DFT

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

 ///////////////////////
 // If-Else Ambiguity //
 ///////////////////////
 //%nonassoc REMOVE_SHIFT_REDUCE_MESSAGE_OF_IF_ELSE_AMBIGUITY
 //%precedence REMOVE_SHIFT_REDUCE_MESSAGE_OF_IF_ELSE_AMBIGUITY
 //%nonassoc ELSE
 //%precedence ELSE
 //%right THEN ELSE
%expect 1 // shift/reduce conflicts

 //////////////////////
 // Specific options //
 //////////////////////
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
nabla_inputstream
: nabla_grammar {rhs;}
| nabla_inputstream nabla_grammar {astAddChild($1,$2);}
;

///////////////////////////
// ∇ scopes: std & std+@ //
///////////////////////////
start_scope: '{' {rhs;};
end_scope: '}' {rhs;} | '}' AT at_constant {rhs;};

////////////////////
// GNU Attributes //
////////////////////
attribute_specifiers
: attribute_specifier {rhs;}
| asm_specifier {rhs;}
| attribute_specifiers attribute_specifier {rhs;}
;
attribute_specifier
: ATTRIBUTE '(' '(' attribute_list ')' ')' {rhs;}
;
attribute_list
: attribute {rhs;}
| attribute_list '(' attribute_list ')' {rhs;}
| attribute_list ','  attribute {rhs;}
;
attribute
: IDENTIFIER {rhs;}
| Z_CONSTANT {rhs;}
| storage_class_specifier {rhs;}
;

/////////////
// GNU ASM //
/////////////
asm_specifier
: ASM '(' asm_list ')' {rhs;}
;
asm_list
: asm_code {rhs;}
| asm_list asm_code {rhs;}
;
asm_code
: STRING_LITERAL {rhs;}
;

//////////////////////////////////////////////////
// ∇ types, qualifiers, specifier, lists & name //
//////////////////////////////////////////////////
type_specifier
: VOID {rhs;}
| CHAR {rhs;} 
| SHORT {rhs;}
| INT {rhs;}
| LONG {rhs;}
| FLOAT {rhs;}
| DOUBLE {rhs;}
| SIGNED {rhs;}
| UNSIGNED {rhs;}
| struct_or_union_specifier {rhs;}
| enum_specifier {rhs;}
| TYPEDEF_NAME {rhs;}		/* after it has been defined as such */
| BOOL {rhs;}
| SIZE_T {rhs;}
| REAL { if (type_precise) preciseY1($$,GMP_REAL) else {rhs;}; type_precise=type_volatile=false;}
| NATURAL {rhs;}
| INTEGER {
    if (type_precise){
      if (type_volatile) volatilePreciseY1($$,GMP_INTEGER)
      else preciseY1($$,GMP_INTEGER)
    }else {rhs;}
    type_precise=type_volatile=false;
  }
| INT32 {rhs;}
| INT64 {rhs;}
| REAL2 {rhs;}
| REAL3 {rhs;}
| REAL3x3 {rhs;}
| REAL2x2 {rhs;}
| CELLTYPE {rhs;}
| NODETYPE {rhs;}
| PARTICLETYPE {rhs;}
| FACETYPE {rhs;}
| UIDTYPE {rhs;}
| FILETYPE {rhs;} 
| OFSTREAM {rhs;} 
| FILECALL '(' IDENTIFIER ',' IDENTIFIER ')' {rhs;} 
;

storage_class_specifier 
: TYPEDEF {rhs;}	/* identifiers must be flagged as TYPEDEF_NAME */
| EXTERN {rhs;}
| STATIC {rhs;}
| AUTO {rhs;}
| INLINE {rhs;}
| REGISTER {rhs;}
;
type_qualifier
: CONST {rhs;}
| ALIGNED {rhs;}
| VOLATILE {{rhs;};type_volatile=true;}
| GMP_PRECISE {{rhs;};type_precise=true;}
;
type_qualifier_list
: type_qualifier {rhs;}
| type_qualifier_list type_qualifier {rhs;}
;
specifier_qualifier_list
: type_specifier specifier_qualifier_list {rhs;}
| type_specifier {rhs;}
| type_qualifier specifier_qualifier_list {rhs;}
| type_qualifier {rhs;}
;
type_name
: specifier_qualifier_list abstract_declarator {rhs;}
| specifier_qualifier_list {rhs;}
;

///////////////////////////////////////////////////////////
// ∇ item(s), group, region, family & system definitions //
///////////////////////////////////////////////////////////
nabla_item
: CELL {rhs;}
| NODE {rhs;}
| FACE {rhs;}
| PARTICLE {rhs;}
;
nabla_items
: CELLS {rhs;}
| NODES {rhs;}
| FACES {rhs;}
| GLOBAL {rhs;}
| PARTICLES {rhs;}
;
nabla_scope: OWN {rhs;} | ALL {rhs;};
nabla_region: INNER {rhs;} | OUTER {rhs;};
nabla_family
: nabla_items {rhs;}
| nabla_scope nabla_items {rhs;}
| nabla_region nabla_items {rhs;}
| nabla_scope nabla_region nabla_items {rhs;}
;
nabla_set
: nabla_family {rhs;}
| nabla_set '+' nabla_family {rhs;}
;

nabla_system
: LID {rhs;}
| SID {rhs;}
| UID {rhs;}
| THIS {rhs;}
| NBNODE {rhs;}
| NBCELL {rhs;}
| BOUNDARY_CELL {rhs;}
| BACKCELL {rhs;}
| BACKCELLUID {rhs;}
| FRONTCELL {rhs;}
| FRONTCELLUID {rhs;}
| NEXTCELL {rhs;}
| NEXTCELL_X {rhs;}
| NEXTCELL_Y {rhs;}
| NEXTCELL_Z {rhs;}
| PREVCELL {rhs;}
| PREVCELL_X {rhs;}
| PREVCELL_Y {rhs;}
| PREVCELL_Z {rhs;}
| NEXTNODE {rhs;}
| PREVNODE {rhs;}
| PREVLEFT {rhs;}
| PREVRIGHT {rhs;}
| NEXTLEFT {rhs;}
| NEXTRIGHT {rhs;}
| TIME {rhs;}
| TIME REMAIN {remainY1();}
| TIME LIMIT {limitY1($$);}
| EXIT {rhs;}
| ITERATION {rhs;}
| MAIL {rhs;}
;

//////////////
// Pointers //
//////////////
pointer
: '*' type_qualifier_list pointer {rhs;}
| '*' type_qualifier_list {rhs;}
| '*' RESTRICT {rhs;}
| '*' pointer {rhs;}
| '*' {rhs;}
;

//////////////////
// INITIALIZERS //
//////////////////
initializer
: assignment_expression {rhs;}
| '{' initializer_list '}' {rhs;}
;
initializer_list
: initializer {rhs;}
| initializer_list ',' initializer {rhs;}
;

//////////////////
// PREPROCESSOR //
//////////////////
preproc
: PREPROCS {
  {rhs;};
  if (sscanf($1->token, "# %d \"%[^\"]\"", &yylineno, nabla_input_file)!=2)
    //nabla_error("declaration sscanf error!");
    nablaError("%s:%d:\n",nabla_input_file,yylineno);
  }
;

//////////////////
// DeclaraTIONS //
//////////////////
declaration
: declaration_specifiers ';' {rhs;}
| declaration_specifiers init_declarator_list ';' {rhs;
    if (typedef_hit){
      //dbg("\n\t[direct_declarator] last_ident is %s", last_identifier);
      dbg("\n\t[direct_declarator] IDENTIFIER: %s => new type!", last_identifier);
      // On créée le nouveau type 
      nablaType *new_type=nMiddleTypeNew();
      new_type->name=strdup(last_identifier);
      // On le rajoute aux connus
      typedef_names=nMiddleTypeAdd(typedef_names,new_type);
      typedef_hit=false;
    }
  }  
;

declaration_specifiers
: storage_class_specifier declaration_specifiers{rhs;}
| storage_class_specifier {rhs;}
| type_specifier declaration_specifiers {rhs;}
| type_specifier {rhs;}
| type_qualifier declaration_specifiers {rhs;}
| type_qualifier {rhs;}
;

/////////////////
// DeclaraTORS //
/////////////////
declarator
: pointer direct_declarator {rhs;}
| direct_declarator {rhs;}
;
identifier_list
: IDENTIFIER {rhs;}
| identifier_list ',' IDENTIFIER	{rhs;}
;
direct_declarator
: IDENTIFIER {rhs;}
| IDENTIFIER SUPERSCRIPT_N_PLUS_ONE {superNP1($$,$1);}
| '(' declarator ')' {rhs;}
| direct_declarator '(' STRING_LITERAL ')' {rhs;}
| direct_declarator '[' constant_expression ']' {rhs;}
| direct_declarator '[' ']' {rhs;}
| direct_declarator '(' parameter_type_list ')' {rhs;}
| direct_declarator '(' parameter_type_list ')' attribute_specifiers {rhs;}
| direct_declarator '(' identifier_list ')' {rhs;}
| direct_declarator '(' identifier_list ')' attribute_specifiers {rhs;}
| direct_declarator '(' ')' {rhs;}
;
init_declarator
:	declarator {rhs;}
|	declarator '=' initializer {rhs;}
;
init_declarator_list
:	init_declarator {rhs;}
|	init_declarator_list ',' init_declarator {rhs;}
;
abstract_declarator
: pointer direct_abstract_declarator {rhs;}
| pointer {rhs;}
| direct_abstract_declarator  {rhs;}
;
direct_abstract_declarator
: '(' abstract_declarator ')' {rhs;}
| '[' ']' {rhs;}
| '[' '*' ']' {rhs;}
| '[' constant_expression ']' {rhs;}
| direct_abstract_declarator '[' ']' {rhs;}
| direct_abstract_declarator '[' constant_expression ']' {rhs;}
| '(' ')'{rhs;}
| '(' parameter_type_list ')' {rhs;}
| direct_abstract_declarator '(' ')' {rhs;}
| direct_abstract_declarator '(' parameter_type_list ')' {rhs;}
;

struct_declarator
: declarator {rhs;}
| ':' constant_expression {rhs;}
| declarator ':' constant_expression {rhs;}
;
struct_declarator_list
: struct_declarator {rhs;}
|	struct_declarator_list ',' struct_declarator {rhs;}
;

////////////////////
// Std parameters //
////////////////////
parameter_type_list
:	parameter_list {rhs;}
|	parameter_list ',' ELLIPSIS {rhs;}
;
parameter_list
:	parameter_declaration {rhs;}
|	parameter_list ',' parameter_declaration {rhs;}
;
parameter_declaration
: nabla_xyz_declaration {rhs;}
| declaration_specifiers declarator {rhs;}
| declaration_specifiers abstract_declarator {rhs;}
| declaration_specifiers {rhs;}
//| GNU_VA_LIST {rhs;}
;

//////////////////////
// ∇ xyz parameters //
//////////////////////
nabla_xyz_direction:IDENTIFIER {rhs;};
nabla_xyz_declaration
: XYZ nabla_xyz_direction {rhs;}
| XYZ nabla_xyz_direction '=' unary_expression{rhs;}
;

/////////////////////////
// ∇ IN/OUT parameters //
/////////////////////////
nabla_inout
: IN {rhs;};
| OUT {rhs;};
| INOUT {rhs;};
;
nabla_parameter_declaration
: direct_declarator {rhs;}
| nabla_item direct_declarator {rhs;}
;
nabla_parameter_declaration_list
: nabla_parameter_declaration {rhs;}
| nabla_parameter_declaration_list ',' nabla_parameter_declaration {rhs;}
;
nabla_inout_parameter
: nabla_inout '(' nabla_parameter_declaration_list ')' {rhs;}
;
nabla_parameter_list
: nabla_inout_parameter {rhs;}
| nabla_parameter_list nabla_inout_parameter {rhs;}
;

//////////////////////////////////
// Arguments of a function call //
//////////////////////////////////
argument_expression_list
: assignment_expression {rhs;}
| argument_expression_list ',' assignment_expression {rhs;}
;

/////////////////
// EXPRESSIONS //
/////////////////
primary_expression
: IDENTIFIER {rhs;}
| IDENTIFIER SUPERSCRIPT_N_PLUS_ONE {superNP1($$,$1);}
| BUILTIN_INFF {rhs;}
| FRACTION_ONE_HALF_CST {rhs;}
| FRACTION_ONE_THIRD_CST {rhs;}
| FRACTION_ONE_QUARTER_CST {rhs;}
| FRACTION_ONE_EIGHTH_CST {rhs;};
| HEX_CONSTANT {rhs;} 
| OCT_CONSTANT {rhs;}
| Z_CONSTANT {rhs;}
| R_CONSTANT {rhs;}
| DIESE {rhs;}
| nabla_item {rhs;}
| nabla_system {rhs;}
| QUOTE_LITERAL {rhs;}
| STRING_LITERAL {rhs;}
| '(' expression ')'	{rhs;}
;
postfix_expression
: primary_expression {rhs;}
| postfix_expression FORALL_NODE_INDEX {rhs;}
| postfix_expression FORALL_CELL_INDEX {rhs;}
| postfix_expression FORALL_MTRL_INDEX 
| postfix_expression '[' expression ']' {rhs;}
| REAL '(' ')'{rhs;}
| REAL '(' expression ')' {rhs;}
| REAL3 '(' ')'{rhs;}
| REAL3 '(' expression ')' {rhs;}
| REAL3x3 '(' ')'{rhs;}
| REAL3x3 '(' expression ')' {rhs;}
| postfix_expression '(' ')' {rhs;}
// On traite l'appel à fatal différemment qu'un CALL standard
| FATAL '(' argument_expression_list ')' {rhs;}
| postfix_expression '(' argument_expression_list ')'{
  // On rajoute un noeud pour annoncer qu'il faut peut-être faire quelque chose lors de l'appel à la fonction
  astNode *callNode=astNewNode();
  // On DOIT laisser un token != NULL!
  callNode->token=strdup("/*call*/");
  callNode->tokenid=CALL;
  astNode *argsNode=astNewNode();
  argsNode->token=strdup("/*args*/");
  argsNode->tokenid=END_OF_CALL;
  RHS($$,callNode,$1,$2,$3,argsNode,$4);
  }
| postfix_expression '.' IDENTIFIER {rhs;}
| postfix_expression '.' nabla_item '(' Z_CONSTANT ')'{rhs;}
| postfix_expression '.' nabla_system {rhs;}
| postfix_expression PTR_OP primary_expression {rhs;} 
| postfix_expression INC_OP {rhs;}
| postfix_expression DEC_OP {rhs;}
| postfix_expression SUPERSCRIPT_DIGIT_TWO {Ypow($$,$1,2);}
| postfix_expression SUPERSCRIPT_DIGIT_THREE {Ypow($$,$1,3);}
//| mathlinks
| aleph_expression
;

///////////////////////////////////
// Unaries (operator,expression) //
///////////////////////////////////
unary_prefix_operator: CENTER_DOT_OP | '*' | '+' | '-' | '~' | '!';
unary_expression
: postfix_expression {rhs;}
| SQUARE_ROOT_OP unary_expression {rhs;}
| CUBE_ROOT_OP unary_expression {rhs;}
| INC_OP unary_expression {rhs;}
| DEC_OP unary_expression {rhs;}
| '&' unary_expression {Yadrs($$,$1,$2);}
| unary_prefix_operator cast_expression {rhs;}
| SIZEOF unary_expression {rhs;}
| SIZEOF '(' type_name ')'{rhs;}
;
cast_expression
: unary_expression {rhs;}
| '(' type_name ')' cast_expression {rhs;}
;
multiplicative_expression
: cast_expression {rhs;}
| multiplicative_expression '*' cast_expression {Yop3p($$,$1,$2,$3);}
| multiplicative_expression '/' cast_expression {Yop3p($$,$1,$2,$3);}
| multiplicative_expression '%' cast_expression {Yop3p($$,$1,$2,$3);}
| multiplicative_expression CROSS_OP cast_expression {Yop3p($$,$1,$2,$3);}
| multiplicative_expression CROSS_OP_2D cast_expression {Yop3p($$,$1,$2,$3);}
| multiplicative_expression CENTER_DOT_OP cast_expression {Yop3p($$,$1,$2,$3);}
| multiplicative_expression CIRCLED_TIMES_OP cast_expression {Yop3p($$,$1,$2,$3);}
| multiplicative_expression CIRCLED_ASTERISK_OP cast_expression {Yop3p($$,$1,$2,$3);}
| multiplicative_expression N_ARY_CIRCLED_TIMES_OP cast_expression {Yop3p($$,$1,$2,$3);}
;
additive_expression
: multiplicative_expression {rhs;}
| additive_expression '+' multiplicative_expression {Yop3p($$,$1,$2,$3);}
| additive_expression '-' multiplicative_expression {Yop3p($$,$1,$2,$3);}
;
shift_expression
: additive_expression {rhs;}
| shift_expression LSH_OP additive_expression {rhs;}
| shift_expression RSH_OP additive_expression {rhs;}
;
relational_expression
: shift_expression {rhs;}
| relational_expression '<' shift_expression {rhs;}
| relational_expression '>' shift_expression {rhs;}
| relational_expression LEQ_OP shift_expression {rhs;}
| relational_expression GEQ_OP shift_expression {rhs;}
;
equality_expression
: relational_expression {rhs;}
| equality_expression EEQ_OP relational_expression {rhs;}
| equality_expression NEQ_OP relational_expression {rhs;}
;
is_test
: IS OWN {rhs;}
| IS INNER {rhs;}
| IS OUTER {rhs;}
| IS IDENTIFIER {rhs;}
;
is_expression
: equality_expression {rhs;}
| is_expression is_test {YopYop(IS_OP_INI,IS_OP_END);}
;
and_expression
: is_expression {rhs;}
| and_expression '&' is_expression {rhs;}
;
exclusive_or_expression
: and_expression {rhs;}
| exclusive_or_expression '^' and_expression {rhs;}
;
inclusive_or_expression
: exclusive_or_expression {rhs;}
| inclusive_or_expression '|' exclusive_or_expression {rhs;}
;
logical_and_expression
: inclusive_or_expression {rhs;}
| logical_and_expression AND_OP inclusive_or_expression {rhs;}
;
logical_or_expression
: logical_and_expression {rhs;}
| logical_or_expression IOR_OP logical_and_expression {rhs;}
;
conditional_expression
: logical_or_expression {rhs;}
| logical_or_expression '?' expression ':' conditional_expression {YopTernary5p($$,$1,$2,$3,$4,$5);}
;

///////////////////////////////////////
// Assignments (operator,expression) //
///////////////////////////////////////
assignment_operator
:  '=' {rhs;}
| MUL_ASSIGN {rhs;} | DIV_ASSIGN {rhs;} | MOD_ASSIGN {rhs;}
| ADD_ASSIGN {rhs;} | SUB_ASSIGN {rhs;}
| LSH_ASSIGN {rhs;} | RSH_ASSIGN {rhs;}
| AND_ASSIGN {rhs;} | XOR_ASSIGN {rhs;} | IOR_ASSIGN {rhs;}
| NULL_ASSIGN {rhs;}
| MIN_ASSIGN {rhs;}
| MAX_ASSIGN {rhs;}
;
assignment_expression
: conditional_expression {rhs;}
| unary_expression assignment_operator assignment_expression {rhs;}
| unary_expression assignment_operator logical_or_expression '?' expression {YopDuaryExpression($$,$1,$2,$3,$5);}
;

expression
: assignment_expression {rhs;}
| expression ',' assignment_expression {rhs;}
;
constant_expression
: conditional_expression {rhs;}
;

////////////////
// Statements //
////////////////
compound_statement
: start_scope end_scope {rhs;}
| start_scope statement_list end_scope {rhs;}
//| start_scope declaration_list end_scope {rhs;}
//| start_scope declaration_list statement_list end_scope {rhs;}
//| start_scope statement_list declaration_list statement_list end_scope {rhs;}
;
expression_statement
: ';' {rhs;}
| expression ';' {rhs;}
;
selection_statement
//: IF '(' expression ')' statement %prec {rhs;}
//: IF '(' expression ')' statement REMOVE_SHIFT_REDUCE_MESSAGE_OF_IF_ELSE_AMBIGUITY {rhs;}
: IF '(' expression ')' statement {rhs;}
| IF '(' expression ')' statement ELSE statement {rhs;}
;

iteration_statement
: FORALL nabla_item statement {forall;}
| FORALL nabla_item AT at_constant statement {forall;}
| FORALL IDENTIFIER CELL statement {forall;}
| FORALL IDENTIFIER NODE statement {forall;}
| FORALL IDENTIFIER FACE statement {forall;}
| FORALL IDENTIFIER PARTICLE statement {forall;}
| WHILE '(' expression ')' statement {rhs;}
| DO statement WHILE '(' expression ')' ';' {rhs;}
| FOR '(' expression_statement expression_statement ')' statement {rhs;}
| FOR '(' expression_statement expression_statement expression ')' statement {rhs;}
| FOR '(' type_specifier expression_statement expression_statement ')' statement {rhs;}
| FOR '(' type_specifier expression_statement expression_statement expression ')' statement {rhs;}
;
jump_statement
: CONTINUE ';' {rhs;}
| BREAK ';' {rhs;}
| RETURN ';' {rhs;}
| RETURN expression ';' {rhs;}
;
statement
: compound_statement {rhs;}
| expression_statement {rhs;}
| selection_statement {rhs;}
| iteration_statement {rhs;}
| jump_statement {rhs;}
| declaration {rhs;}
;
statement_list
: statement {rhs;}
| statement_list statement {rhs;}
;

/////////////////
// ∇ functions //
/////////////////
function_definition
: declaration_specifiers declarator compound_statement {rhs;}
| declaration_specifiers declarator AT at_constant compound_statement {rhs;}
;

/////////////////////////
// ∇ items definitions //
/////////////////////////
nabla_item_definition
: nabla_items '{' nabla_item_declaration_list '}' ';' {rhs;}
;
nabla_item_declaration_list
: nabla_item_declaration {rhs;}
| nabla_item_declaration_list nabla_item_declaration {rhs;}
;
nabla_item_declaration
: type_name nabla_direct_declarator_list ';' {rhs;}
| preproc {rhs;}
;
nabla_direct_declarator_list
: nabla_direct_declarator {rhs;}
| nabla_direct_declarator_list ',' nabla_direct_declarator {rhs;}
;
nabla_direct_declarator
: IDENTIFIER {rhs;}
| IDENTIFIER '[' nabla_items ']'{rhs;}
| IDENTIFIER '[' primary_expression ']'{rhs;}
| IDENTIFIER  SUPERSCRIPT_N_PLUS_ONE {superNP1($$,$1);}
;

//////////////////////////
// ∇ options definition //
//////////////////////////
nabla_options_definition
: OPTIONS '{' nabla_option_declaration_list '}' ';' {rhs;}
nabla_option_declaration_list
: nabla_option_declaration {rhs;}
| nabla_option_declaration_list nabla_option_declaration {rhs;};
nabla_option_declaration
: type_specifier direct_declarator ';' {rhs;}
| type_specifier direct_declarator '=' expression ';' {rhs;}  
| preproc {rhs;}
;

///////////////////////
// ∇ '@' definitions //
///////////////////////
at_single_constant
: Z_CONSTANT {rhs;}
| R_CONSTANT {rhs;}
| '-' Z_CONSTANT {rhs;}
| '+' Z_CONSTANT {rhs;}
| '-' R_CONSTANT {rhs;}
| '+' R_CONSTANT {rhs;}
;
at_tree_constant
: at_single_constant {rhs;}
| at_tree_constant '/' at_single_constant {rhs;}
;
at_constant
// On rajoute des parenthèses (Yp1p),
// que l'on enlevera lors du DFS nMiddleAtConstantParse
: at_tree_constant {Yp1p($$,$1);} 
| at_constant ',' at_tree_constant {Yp3p($$,$1,$2,$3);};

////////////////////////
// ∇ jobs definitions //
////////////////////////
org_mode_stars
: '*' {rhs;}
| org_mode_stars '*' {rhs;}
;

nabla_job_prefix
: FORALL nabla_set {rhs;}
| org_mode_stars FORALL nabla_set {rhs;}
;
nabla_job_decl
: nabla_job_prefix {rhs;}
| nabla_job_prefix declaration_specifiers IDENTIFIER '(' parameter_type_list ')' {rhs;}
;

nabla_job_definition
: nabla_job_decl ',' unary_expression assignment_operator assignment_expression ';' {job;}
| nabla_job_decl compound_statement {job;}
| nabla_job_decl nabla_parameter_list compound_statement {job;}
| nabla_job_decl AT at_constant compound_statement {job;}
| nabla_job_decl AT at_constant IF '(' constant_expression ')' compound_statement {job;}
| nabla_job_decl nabla_parameter_list AT at_constant compound_statement {job;}
| nabla_job_decl nabla_parameter_list AT at_constant IF '(' constant_expression ')' compound_statement {job;}
;

////////////////////////
// ∇ single reduction //
////////////////////////
nabla_reduction
: nabla_job_prefix IDENTIFIER MIN_ASSIGN IDENTIFIER  AT at_constant ';' {rhs;}
| nabla_job_prefix IDENTIFIER MAX_ASSIGN IDENTIFIER  AT at_constant ';' {rhs;}
;

/////////////////
// ∇ libraries //
/////////////////
single_library:
  LIB_DFT         {rhs;}
| LIB_GMP         {rhs;}
| LIB_MPI         {rhs;}
| MAIL            {rhs;}
| PARTICLES       {rhs;}
| LIB_ALEPH       {rhs;}
| LIB_SLURM       {rhs;}
| LIB_CARTESIAN   {rhs;}
| LIB_MATHEMATICA {rhs;}
| REAL {rhs;}
| REAL2 {rhs;}
;
with_library_list
  : single_library
| with_library_list ',' single_library {rhs;}
;
with_library: WITH with_library_list ';'{rhs;};

///////////////
// ∇ grammar //
///////////////
nabla_grammar
// On patche l'espace qui nous a été laissé par le sed pour remettre le bon '#'include
: INCLUDES {$1->token[0]='#';   {rhs;}}
| preproc                       {rhs;}
| with_library                  {rhs;}
| declaration                   {rhs;}
| nabla_options_definition      {rhs;}
| nabla_item_definition         {rhs;}
| function_definition	        {rhs;}
| nabla_job_definition          {rhs;}
| nabla_reduction               {rhs;}
;

///////////////////////
// Aleph Expressions //
///////////////////////
aleph_vector
: ALEPH_RHS {rhs;}
| ALEPH_LHS {rhs;}
;

aleph_expression
: aleph_vector {rhs;} // Utilisé pour dumper par exemple
| LIB_ALEPH aleph_vector ALEPH_RESET {rhs;}
| LIB_ALEPH ALEPH_SOLVE {rhs;}
| LIB_ALEPH aleph_vector ALEPH_NEW_VALUE {rhs;}
| LIB_ALEPH aleph_vector ALEPH_ADD_VALUE {rhs;}
| LIB_ALEPH aleph_vector ALEPH_SET {rhs;}
| LIB_ALEPH ALEPH_MTX ALEPH_ADD_VALUE {rhs;}
| LIB_ALEPH ALEPH_MTX ALEPH_SET {rhs;}
| LIB_ALEPH ALEPH_LHS ALEPH_GET {rhs;}
| LIB_ALEPH ALEPH_RHS ALEPH_GET {rhs;}
;

/////////////////////////////
// STRUCTS, ENUMS & UNIONS //
/////////////////////////////
struct_declaration_list
: struct_declaration_list preproc{rhs;}
| struct_declaration {rhs;}
| struct_declaration_list struct_declaration {rhs;}
;
struct_declaration
: specifier_qualifier_list struct_declarator_list ';' {rhs;}
;

// ENUMERATORS
enumerator
: IDENTIFIER {rhs;}
| IDENTIFIER '=' constant_expression {rhs;}
;
enumerator_list
: enumerator {rhs;}
| enumerator_list ',' enumerator {rhs;}
;
enum_specifier
: ENUM '{' enumerator_list '}' {rhs;}
| ENUM IDENTIFIER '{' enumerator_list '}' {rhs;}
| ENUM IDENTIFIER {rhs;}
;

// SPECIFIERS
struct_or_union
: STRUCT {rhs;}
| UNION {rhs;}
;

// Structs or Unions
struct_or_union_specifier
: struct_or_union IDENTIFIER '{' struct_declaration_list '}' {rhs;}
| struct_or_union TYPEDEF_NAME '{' struct_declaration_list '}' {rhs;}
| struct_or_union '{' struct_declaration_list '}' {rhs;}
| struct_or_union TYPEDEF_NAME {rhs;}
| struct_or_union IDENTIFIER {rhs;
    dbg("\n[struct_or_union_specifier] Adding *struct* new type '%s'",last_identifier);
    // On créée le nouveau type 
    nablaType *new_type=nMiddleTypeNew();
    new_type->name=strdup(last_identifier);
    // On le rajoute aux connus
    typedef_names=nMiddleTypeAdd(typedef_names,new_type);
  }
;


%%

// ****************************************************************************
// * tokenidToRuleid
// *****************************************************************************
inline int tokenidToRuleid(int tokenid){
  return YYTRANSLATE(tokenid);
}


// ****************************************************************************
// * yyTranslate
// ****************************************************************************
inline int yyTranslate(int tokenid){
  return YYTRANSLATE(tokenid);
}


// ****************************************************************************
// * yyUndefTok
// ****************************************************************************
inline int yyUndefTok(void){
  return YYUNDEFTOK;
}


// ****************************************************************************
// * yyNameTranslate
// ****************************************************************************
inline int yyNameTranslate(int tokenid){
  return yytname[YYTRANSLATE(tokenid)][1];
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


// ****************************************************************************
// * tokenToId
// ****************************************************************************
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
// * Standard rhsTailSandwich
// *****************************************************************************
inline void rhsTailSandwich(astNode **lhs,int yyn,
                            int left_token, int right_token, astNode* *yyvsp){
  // Nombre d'éléments dans notre RHS
  const int yynrhs = yyr2[yyn];
  // Le first est le nouveau noeud que l'on va insérer
  astNode *first=*lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);
  // Le next pointe pour l'instant sur le premier noeud en argument
  astNode *next=yyvsp[(0+1)-(yynrhs)];
  // On prépare les 2 tokens à rajouter
  astNode *left=astNewNode();
  left->token=toolStrQuote(strdup(yytname[YYTRANSLATE(left_token)]));
  left->tokenid=left_token;
  astNode *right=astNewNode();
  right->token=toolStrQuote(strdup(yytname[YYTRANSLATE(right_token)]));
  right->tokenid=right_token;
  // Dans le cas où il n'y en a qu'un, le sandwich est différent:
  if (yynrhs==1){
    astAddChild(first,left);
    astAddNext(left,next);
    astAddNext(next,right);
    return;
  }
  // S'il y en a plus qu'un. on déroule les boucles
  astAddChild(first,next);
  // On saute le premier et s'arrète avant le dernier
  for(int i=1;i<yynrhs-1;i+=1){
    //printf("[1;33m[rhsTailSandwich] \t for\n[m");
    first=next;
    next=yyvsp[(i+1)-(yynrhs)];
    astAddNext(first,next);
  }
   // On récupère le dernier
  astNode *tail=yyvsp[0];
  // Et on sandwich
  astAddNext(next,left);
  astAddNext(left,tail);
  astAddNext(tail,right);
}


// *****************************************************************************
// * Variadic rhsTailSandwich
// *****************************************************************************
inline void rhsTailSandwichVariadic(astNode **lhs,int yyn,int yynrhs,
                                    int left_token, int right_token, ...){
  va_list args;
  va_start(args, right_token);
  //("[1;33m[rhsTailSandwich] yynrhs=%d\n[m",yynrhs);
  // Le first est le nouveau noeud que l'on va insérer
  astNode *first=*lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);
  // Le next pointe pour l'instant sur le premier noeud en argument
  astNode *next=va_arg(args,astNode*);
  // On prépare les 2 tokens à rajouter
  astNode *left=astNewNode();
  left->token=toolStrQuote(strdup(yytname[YYTRANSLATE(left_token)]));
  left->tokenid=left_token;
  astNode *right=astNewNode();
  right->token=toolStrQuote(strdup(yytname[YYTRANSLATE(right_token)]));
  right->tokenid=right_token;
  // Dans le cas où il n'y en a qu'un, le sandwich est différent:
  if (yynrhs==1){
    astAddChild(first,left);
    astAddNext(left,next);
    astAddNext(next,right);
    va_end(args);
    return;
  }
  // S'il y en a plus qu'un. on déroule les boucles
  astAddChild(first,next);
  // On saute le premier et s'arrète avant le dernier
  for(int i=1;i<yynrhs-1;i+=1){
    //printf("[1;33m[rhsTailSandwich] \t for\n[m");
    first=next;
    next=va_arg(args,astNode*);
    astAddNext(first,next);
  }
  // On récupère le dernier
  astNode *tail=va_arg(args,astNode*);
  // Et on sandwich
  astAddNext(next,left);
  astAddNext(left,tail);
  astAddNext(tail,right);
  va_end(args);
  //("[1;33m[rhsTailSandwich] done[m\n");
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


// *****************************************************************************
// * Variadic rhsAdd
// *****************************************************************************
inline void rhsAddVariadic(astNode **lhs,int yyn,int yynrhs,...){
  va_list args;
  assert(yynrhs>0);
  va_start(args, yynrhs);
  //"[rhsAddGeneric] On accroche le nouveau noeud au lhs\n");
  astNode *first=*lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);
  astNode *next=va_arg(args,astNode*);
  assert(next);
  //("[rhsAddGeneric] On commence par rajouter le premier comme fils\n");
  astAddChild(first,next);
  for(int i=1;i<yynrhs;i+=1){
    first=next;
    next=va_arg(args,astNode*);
    astAddNext(first,next);
  }
  va_end(args);
}


// *****************************************************************************
// * rhsYSandwich
// *****************************************************************************
inline void rhsYSandwich(astNode **lhs,int yyn, astNode* *yyvsp,
                         int left_token,
                         int right_token){
  // Nombre d'éléments dans notre RHS
  const int yynrhs = yyr2[yyn];
  // Le first est le nouveau noeud que l'on va insérer
  astNode *first=*lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);
  // On prépare le token de gauche à rajouter
  astNode *left=astNewNode();
  left->token=toolStrQuote(strdup(yytname[YYTRANSLATE(left_token)]));
  left->tokenid=left_token;
  // Le next pointe pour l'instant sur le premier noeud en argument
  astNode *next=yyvsp[(0+1)-(yynrhs)];
  // On prépare le token de droite à rajouter
  astNode *right=astNewNode();
  right->token=toolStrQuote(strdup(yytname[YYTRANSLATE(right_token)]));
  right->tokenid=right_token;
  // Dans le cas où il n'y en a qu'un, le sandwich est différent:
  if (yynrhs==1){
    astAddChild(first,left);
    astAddNext(left,next);
    astAddNext(next,right);
    return;
  }
  // S'il y en a plus qu'un, on déroule les boucles
  astAddChild(first,left);
  astAddNext(left,next);
  // On saute le premier et s'arrète avant le dernier
  for(int i=1;i<yynrhs-1;i+=1){
    //printf("[1;33m[rhsYSandwich] \t for\n[m");
    first=next;
    next=yyvsp[(i+1)-(yynrhs)];
    astAddNext(first,next);
  }
  // On récupère le dernier
  astNode *tail=yyvsp[0];
  // Et on sandwich
  astAddNext(next,tail);
  astAddNext(tail,right);
  //printf("[1;33m[rhsYSandwich] done[m\n");
}


  
// *****************************************************************************
// * Variadic rhsYSandwich
// *****************************************************************************
inline void rhsYSandwichVariadic(astNode **lhs,int yyn,int yynrhs,
                                 int left_token,
                                 int right_token, ...){
  va_list args;
  va_start(args, right_token);
  //printf("[1;33m[rhsYSandwich] yynrhs=%d\n[m",yynrhs);
  // Le first est le nouveau noeud que l'on va insérer
  astNode *first=*lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);
  // On prépare le token de gauche à rajouter
  astNode *left=astNewNode();
  left->token=toolStrQuote(strdup(yytname[YYTRANSLATE(left_token)]));
  left->tokenid=left_token;
  // Le next pointe pour l'instant sur le premier noeud en argument
  astNode *next=va_arg(args,astNode*);
  // On prépare le token de droite à rajouter
  astNode *right=astNewNode();
  right->token=toolStrQuote(strdup(yytname[YYTRANSLATE(right_token)]));
  right->tokenid=right_token;
  // Dans le cas où il n'y en a qu'un, le sandwich est différent:
  if (yynrhs==1){
    astAddChild(first,left);
    astAddNext(left,next);
    astAddNext(next,right);
    va_end(args);
    return;
  }
  // S'il y en a plus qu'un, on déroule les boucles
  astAddChild(first,left);
  astAddNext(left,next);
  // On saute le premier et s'arrète avant le dernier
  for(int i=1;i<yynrhs-1;i+=1){
    //printf("[1;33m[rhsYSandwich] \t for\n[m");
    first=next;
    next=va_arg(args,astNode*);
    astAddNext(first,next);
  }
  // On récupère le dernier
  astNode *tail=va_arg(args,astNode*);
  // Et on sandwich
  astAddNext(next,tail);
  astAddNext(tail,right);
  va_end(args);  
  //printf("[1;33m[rhsYSandwich] done[m\n");
}
