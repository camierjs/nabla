/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccMiddlend.h     															  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2012.12.13																	  *
 * Updated  : 2013.09.12																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2012.12.13	camierjs	Creation															  *
 *****************************************************************************/
#ifndef _NCC_MIDDLEND_H_
#define _NCC_MIDDLEND_H_

// Enum�ration des libraries possible depuis Arcane
typedef enum {
  mpi=0,
  gmp,
  dft,
  mail,
  aleph,
  slurm,
  particles,
  cartesian,
  materials,
  mathematica,
} library;


// Enum�ration des possibilit�s des variables en in/out/inout
typedef enum {
  enum_in_variable=0,
  enum_out_variable,
  enum_inout_variable
} inout_mode;


// Middlend VARIABLE 
typedef struct nablaVariableStruct{
  bool axl_it;
  char *item; // node, cell, etc.
  char *type; // int, Real, Real3, etc.
  char *name; // son nom
  char *field_name; // son nom
  int gmpRank;   // est-il g�r� par une indirection multi precision? -1 == non, sinon le rank
  int dim;    // sa dimension
  long size;   // sa taille dans le cas o� dim vaut '1'
  bool dump;
  bool is_gathered;
  inout_mode inout;
  struct nablaMainStruct *main;
  struct nablaVariableStruct *next;
}nablaVariable;


// Backend OPTION 
typedef struct nablaOptionStruct{
  bool axl_it;
  const char *type; // int, Real, Real3, etc.
  const char *name; // son nom
  char *dflt; // sa valeur par d�fault
  struct nablaMainStruct *main;
  struct nablaOptionStruct *next;
}nablaOption;


// Middlend JOB 
typedef struct nablaJobStruct{
  bool is_an_entry_point;
  bool is_a_function;
  char *scope;
  char *region;
  char *item;
  char *rtntp;//return_type;
  char *name;
  char *name_utf8;
  char *xyz;
  char *drctn;//direction
  char at[128]; // Buffer de construction
  int whenx; // index des whens
  double whens[32];
  char *where;
  astNode *returnTypeNode;
  astNode *stdParamsNode;
  astNode *nblParamsNode;
  astNode *ifAfterAt;
  nablaVariable *called_variables;
  nablaVariable *in_out_variables;
  nablaVariable *variables_to_gather_scatter;
  char foreach_item;
  struct{
    bool left_of_assignment_operator;
    bool turnBracketsToParentheses;
    char enum_enum;
    bool got_a_return;
    bool got_a_return_and_the_semi_colon;
    int imbricated_scopes;
    int isPostfixed;
    int isDotXYZ;
    bool diffracting;
    astNode *statementToDiffract;
    int diffractingXYZ;
    bool entityScopeToDump;
    char *entityScopeString;
    bool selection_statement_in_compound_statement;
    char *function_call_name;
    bool function_call_arguments;
    bool postfix_constant;
    bool variableIsArray;
    bool alephKeepExpression;
    inout_mode inout; // mode en cours, parmis in, out et inout
    int iGather; // Indice de la variable qui est � gatherer
    int iScatter; // Indice de la variable qui est � scatterer
    // Bool pour savoir si l'on est en train de travailler pour returner depuis
    // un argument du job courant
    bool returnFromArgument; 
  } parse;
  struct nablaEntityStruct *entity;
  struct nablaJobStruct *next;
}nablaJob;
 

// Middlend ENTITY 
typedef struct nablaEntityStruct{
  FILE *hdr, *src;
  const char *name;
  const char *name_upcase;
  nablaJob *jobs;
  library libraries;
  struct nablaMainStruct *main; 
  struct nablaEntityStruct *next;
} nablaEntity;


// Backend HOOKS
typedef struct nablaBackendHooksStruct{
  // Prefix � l'ENUMERATE_*
  char* (*prefixEnumerate)(nablaJob *);
  // Produit l'ENUMERATE_* avec XYZ
  char* (*dumpEnumerateXYZ)(nablaJob *);
  // Dump l'ENUMERATE_*
  char* (*dumpEnumerate)(nablaJob *);
  // Postfix � l'ENUMERATE_*
  char* (*postfixEnumerate)(nablaJob *);
  // Dump la r�f�rence � un item au sein d'un ENUMERATE_*
  char* (*item)(nablaJob*,const char, const char, char);
  // Gestion des diff�rentes actions pour un job
  void (*switchTokens)(astNode *, nablaJob *);
  // Transformation de tokens en variables selon l'ENUMERATE_*
  nablaVariable* (*turnTokenToVariable)(astNode*,struct nablaMainStruct*, nablaJob *);
  void (*system)(astNode * n, struct nablaMainStruct *arc, const char cnf, char enum_enum);
  // Permet de rajouter des param�tres aux fonctions (coords/globals)
  void (*addExtraParameters)(struct nablaMainStruct *nabla, nablaJob*, int *numParams);
  // Dump dans le src des parametres nabla en in comme en out
  // Et dans le cas Okina de remplir quelles variables in on va utiliser pour les gather/scatter
  void (*dumpNablaParameterList)(struct nablaMainStruct*, nablaJob*, astNode*,int *);
  void (*turnBracketsToParentheses)(struct nablaMainStruct*, nablaJob*, nablaVariable*, char);
  // Gestion de l'ex diffraction (plus utilis�)
  void (*diffractStatement)(struct nablaMainStruct *, nablaJob *, astNode **);
  // Hook pour dumper le nom de la fonction
  void (*functionName)(struct nablaMainStruct *);
  // Hook de g�n�ration d'un kernel associ� � une fonction
  void (*function)(struct nablaMainStruct *, astNode *);
  // G�n�ration d'un kernel associ� � un support
  void (*job)(struct nablaMainStruct *, astNode *);
  // Hooks additionnels pour sp�cifier de fa�on propre au backend:
  // le num�ro de l'it�ration, l'appel pour quitter, r�cup�rer le temps de la simulation, etc.
  void (*iteration)(struct nablaMainStruct *);
  void (*exit)(struct nablaMainStruct *);
  void (*time)(struct nablaMainStruct *);
  void (*fatal)(struct nablaMainStruct *);
  // Hooks pour rajouter au fur et � mesure qu'on les d�couvre
  // les fonctions appel�es et les arguments
  void (*addCallNames)(struct nablaMainStruct*,nablaJob*,astNode*);
  void (*addArguments)(struct nablaMainStruct*,nablaJob*);
  // Hook pour mettre en forme les options
  void (*turnTokenToOption)(struct nablaMainStruct*,nablaOption*);
  // Hook pour pr�fixer les points d'entr�e (�-la inline, par exemple)
  char* (*entryPointPrefix)(struct nablaMainStruct*,nablaJob*);
  // Hook pour associer aux fonctions appel�es les arguments � rajouter
  void (*dfsForCalls)(struct nablaMainStruct*,nablaJob*,astNode*,const char *,astNode *);
  // Hook pour transformer les variables � returner
  bool (*primary_expression_to_return)(struct nablaMainStruct*, nablaJob*, astNode*);
  // Hook returnFromArgument for OKINA and OMP
  void (*returnFromArgument)(struct nablaMainStruct*, nablaJob*);
} nablaBackendHooks;

typedef struct nablaDefinesStruct{
  char *what;
  char *with;
}nablaDefine;

typedef struct nablaTypedefStruct{
  char *what;
  char *with;
}nablaTypedef;


// Structure des hooks que l'on va utiliser afin de g�n�rer pour AVX ou MIC
typedef struct nablaBackendSimdHooksStruct{
  char* (*bits)(void);
  char* (*gather)(nablaJob*,nablaVariable*,enum_phase);
  char* (*scatter)(nablaVariable*);
  nablaTypedef* typedefs;
  nablaDefine* defines;
  char** forwards;
  char* (*prevCell)(void);
  char* (*nextCell)(void);
  char* (*includes)(void);
} nablaBackendSimdHooks;


struct nablaMainStruct;

// Structure des hooks que l'on va utiliser afin de g�n�rer avec ou sans parallel color
typedef struct nablaBackendParallelHooksStruct{
  char* (*sync)(void);
  char* (*spawn)(void);
  char* (*loop)(struct nablaMainStruct*);
  char* (*includes)(void);
} nablaBackendParallelHooks;


// Structure des hooks que l'on va utiliser afin de g�n�rer les pragmas
typedef struct nablaBackendPragmaHooksStruct{
  char* (*ivdep)(void);
  char* (*align)(void);
} nablaBackendPragmaHooks;


// Middlend TOP 
typedef struct nablaMainStruct{
  FILE *main, *cfg, *axl, *dot;
  const char *name;
  char *tmpVarKinds;
  nablaVariable *variables;
  nablaOption *options;
  nablaEntity *entity;
  BACKEND_SWITCH backend;
  BACKEND_COLORS colors;
  char *interface_name;
  char *interface_path;
  char *service_name;
  bool optionDumpTree;
  struct nablaBackendHooksStruct *hook;
  struct nablaBackendSimdHooksStruct *simd;
  struct nablaBackendParallelHooksStruct *parallel;
  struct nablaBackendPragmaHooksStruct *pragma;
} nablaMain;


NABLA_STATUS nablaInclude(nablaMain*,char*);
NABLA_STATUS nablaDefines(nablaMain*,nablaDefine*);
NABLA_STATUS nablaTypedefs(nablaMain*,nablaTypedef*);
NABLA_STATUS nablaForwards(nablaMain*,char**);
NABLA_STATUS nablaCompoundJobEnd(nablaMain*);

int nablaMiddlendSwitch(astNode*,
                        const bool,
                        //const char*,
                        const char*,
                        const BACKEND_SWITCH,
                        const BACKEND_COLORS,
                        char*,char*,char*);
void nablaMiddlendParseAndHook(astNode *, nablaMain *);
int dumpParameterTypeList(FILE*, astNode*);

// WHENS
void nablaStoreWhen(nablaMain *, char *);
int nablaComparEntryPoints(const void *one, const void *two);
int nablaNumberOfEntryPoints(nablaMain *);
nablaJob* nablaEntryPointsSort(nablaMain *);

// SPACE
void nablaInsertSpace(nablaMain*,astNode*);

// AT
void nablaAtConstantParse(astNode *, nablaMain *, char *);

// ITEMS
void nablaItems(astNode * n, int ruleid, nablaMain *arc);

// VARIABLES
nablaVariable *nablaVariableNew(nablaMain *);
nablaVariable *nablaVariableAdd(nablaMain*, nablaVariable *);
nablaVariable *nablaVariableLast(nablaVariable *);
nablaVariable *nablaVariableFind(nablaVariable *variables, char *name);
what_to_do_with_the_postfix_expressions nablaVariables(nablaMain *arc,
                                                       astNode * n,
                                                       const char cnf,
                                                       char enum_enum);
int nablaVariableGmpRank(nablaVariable *variables);
char *nablaVariableGmpNameRank(nablaVariable *variables, int k);
bool nablaVariableGmpDumpRank(nablaVariable *variables, int k);
int nablaVariableGmpDumpNumber(nablaVariable *variables);

// OPTIONS
nablaOption *nablaOptionNew(nablaMain *);
nablaOption *nablaOptionLast(nablaOption *) ;
nablaOption *nablaOptionAdd(nablaMain *, nablaOption *);
nablaOption *findOptionName(nablaOption *options, char *);
void nablaOptions(astNode * n, int ruleid, nablaMain *);
nablaOption *turnTokenToOption(astNode*, nablaMain*);

// ENTITY
nablaEntity *nablaEntityNew(nablaMain*);
nablaEntity *nablaEntityAddEntity(nablaMain*, nablaEntity*);

// FUNTIONS
void nablaFunctionDumpHdr(FILE *file, astNode * n);
void nablaFunctionParse(astNode * n, nablaJob *fct);
void nablaFctFill(nablaMain *,
                  nablaJob *,
                  astNode *n,
                  const char *);

// JOBS
void scanForNablaJobParameter(astNode * n, int ruleid, nablaMain *arc);
void scanForNablaJobAtConstant(astNode * n, nablaMain *arc);
char scanForNablaJobForeachItem(astNode*);
nablaJob *nablaJobNew(nablaEntity *);
nablaJob *nablaJobAdd(nablaEntity*, nablaJob*);
nablaJob *nablaJobLast(nablaJob *);
nablaJob *nablaJobFind(nablaJob *,char *);
void nablaJobParse(astNode *, nablaJob *);
void nablaJobFill(nablaMain*, nablaJob*, astNode *, const char *);
void scanForIfAfterAt(astNode *, nablaJob *, nablaMain *);
void dumpIfAfterAt(astNode *, nablaMain *);


// LIBRARIES
void nablaLibraries(astNode *, nablaEntity *);


// Time tree dump
NABLA_STATUS timeTreeSave(nablaMain*, nablaJob*, int);

#endif // _NABLA_MIDDLEND_H_
