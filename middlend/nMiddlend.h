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
#ifndef _NABLA_MIDDLEND_H_
#define _NABLA_MIDDLEND_H_

// Enum�ration des libraries possible depuis Arcane
typedef enum {
  with_mpi=0,
  with_gmp,
  with_dft,
  with_mail,
  with_aleph,
  with_slurm,
  with_particles,
  with_cartesian,
  with_materials,
  with_mathematica,
  with_real          // 1D only, no Real3 admitted
} with_library;


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
  char forall_item;
  // Bool pour savoir si le job est un job de r�duction
  bool reduction;
  char *reduction_name;
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
  with_library libraries;
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
  // G�n�ration d'un kernel associ� � une reduction
  void (*reduction)(struct nablaMainStruct *, astNode *);
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

// nMiddleLibraries.c
bool isWithLibrary(nablaMain*,with_library);
void nMiddleLibraries(astNode*, nablaEntity*);

// nMiddleEntities.c
nablaEntity *nMiddleEntityNew(nablaMain*);
nablaEntity *nMiddleEntityAddEntity(nablaMain*, nablaEntity*);

// nMiddleJobs.c
void nMiddleScanForNablaJobParameter(astNode * n, int ruleid, nablaMain *arc);
void nMiddleScanForNablaJobAtConstant(astNode * n, nablaMain *arc);
char nMiddleScanForNablaJobForallItem(astNode*);
void nMiddleScanForIfAfterAt(astNode *, nablaJob *, nablaMain *);
void nMiddleDumpIfAfterAt(astNode *, nablaMain *);
int nMiddleDumpParameterTypeList(FILE*, astNode*);
nablaJob *nMiddleJobNew(nablaEntity *);
nablaJob *nMiddleJobAdd(nablaEntity*, nablaJob*);
nablaJob *nMiddleJobLast(nablaJob *);
nablaJob *nMiddleJobFind(nablaJob *,char *);
void nMiddleJobParse(astNode *, nablaJob *);
void nMiddleJobFill(nablaMain*, nablaJob*, astNode *, const char *);

// nMiddleVariables.c
nablaVariable *nMiddleVariableNew(nablaMain *);
nablaVariable *nMiddleVariableAdd(nablaMain*, nablaVariable *);
nablaVariable *nMiddleVariableLast(nablaVariable *);
nablaVariable *nMiddleVariableFind(nablaVariable *variables, char *name);
what_to_do_with_the_postfix_expressions nMiddleVariables(nablaMain *arc,
                                                       astNode * n,
                                                       const char cnf,
                                                       char enum_enum);
int nMiddleVariableGmpRank(nablaVariable *variables);
char *nMiddleVariableGmpNameRank(nablaVariable *variables, int k);
bool nMiddleVariableGmpDumpRank(nablaVariable *variables, int k);
int nMiddleVariableGmpDumpNumber(nablaVariable *variables);

// nMiddleOptions.c
nablaOption *nMiddleOptionNew(nablaMain*);
nablaOption *nMiddleOptionLast(nablaOption*) ;
nablaOption *nMiddleOptionAdd(nablaMain*,nablaOption*);
nablaOption *nMiddleOptionFindName(nablaOption *options, char *);
void nMiddleOptions(astNode * n, int ruleid, nablaMain *);
nablaOption *nMiddleTurnTokenToOption(astNode*, nablaMain*);

// nMiddle.c
NABLA_STATUS nMiddleCompoundJobEnd(nablaMain*);
NABLA_STATUS nMiddleInclude(nablaMain*,char*);
NABLA_STATUS nMiddleDefines(nablaMain*,nablaDefine*);
NABLA_STATUS nMiddleTypedefs(nablaMain*,nablaTypedef*);
NABLA_STATUS nMiddleForwards(nablaMain*,char**);
void nMiddleParseAndHook(astNode*,nablaMain*);
void nMiddleInsertSpace(nablaMain*,astNode*);
int nMiddleSwitch(astNode*,const bool,const char*,
                  const BACKEND_SWITCH,
                  const BACKEND_COLORS,
                  char*,char*,char*);
// nMiddlePrintf
int nprintf(const struct nablaMainStruct *nabla, const char *debug, const char *format, ...);
int hprintf(const struct nablaMainStruct *nabla, const char *debug, const char *format, ...);

// nMiddleItems
void nMiddleItems(astNode * n, int ruleid, nablaMain *arc);

// nMiddleHLT: @ + When[s]
void nMiddleAtConstantParse(astNode *, nablaMain *, char *);
void nMiddleStoreWhen(nablaMain *, char *);
int nMiddleComparEntryPoints(const void *one, const void *two);
int nMiddleNumberOfEntryPoints(nablaMain *);
nablaJob* nMiddleEntryPointsSort(nablaMain *,int);

// nMiddleTimeTree.c
NABLA_STATUS nMiddleTimeTreeSave(nablaMain*, nablaJob*, int);

// nMiddleFunctions
void nMiddleFunctionDumpHeader(FILE*,astNode*);
void nMiddleFunctionParse(astNode*,nablaJob*);
void nMiddleFunctionFill(nablaMain*,nablaJob*,astNode*,const char*);

#endif // _NABLA_MIDDLEND_H_