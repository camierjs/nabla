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
#ifndef _NABLA_MIDDLEND_H_
#define _NABLA_MIDDLEND_H_


// ****************************************************************************
// * Enumération des directions possibles
// ****************************************************************************
typedef enum {
  DIR_UNKNOWN,
  DIR_X,
  DIR_Y,
  DIR_Z
} dir_xyz;

  
// ****************************************************************************
// * Enumération des libraries possibles
// ****************************************************************************
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
  with_real,          // 1D only, no Real3 admitted
  with_real2          // 2D only, no Real3 admitted
} enum_library;


// ****************************************************************************
// Enumération des possibilités des variables en in/out/inout
// ****************************************************************************
typedef enum {
  enum_in_variable=0,
  enum_out_variable,
  enum_inout_variable
} inout_mode;


// ****************************************************************************
// * Standard Association Struct
// ****************************************************************************
typedef struct nWhatWithStruct{
  char *what;
  char *with;
} nWhatWith;


// ****************************************************************************
// * Nabla Grammar TYPE struct
// ****************************************************************************
typedef struct nablaTypeStruct{
  const char *name;
  struct nablaTypeStruct *next;
}nablaType;


// ****************************************************************************
// * Nabla OPTION struct
// ****************************************************************************
typedef struct nablaOptionStruct{
  char *type; // int, Real, Real3, etc.
  char *name; // son nom
  char *dflt; // sa valeur par défault
  bool axl_it;
  struct nablaMainStruct *main;
  struct nablaOptionStruct *next;
}nablaOption;


// ****************************************************************************
// * PowerType struct
// * Soit on a un ID direct sous le power_dimension
// * Soit l'ID est le nom de la function et les args les ID des dimensions
// ****************************************************************************
typedef struct nablaPowerTypeStruct{
  char* id;
  size_t nargs;
  char* *args;
  struct nablaPowerTypeStruct *next;
}nablaPowerType;


// ****************************************************************************
// * Nabla VARIABLE struct
// ****************************************************************************
typedef struct nablaVariableStruct{
  char *item; // node, cell, etc.
  char *type; // int, Real, Real3, etc.
  char *name; // son nom
  char *field_name; // son nom
  int gmpRank;   // est-il géré par une indirection multi precision? -1 == non, sinon le rank
  int dim;    // sa dimension
  long size;  // sa taille dans le cas où dim vaut '1'
  bool dump;
  bool in;
  bool out;
  bool is_gathered;
  inout_mode inout;
  bool axl_it;
  struct nablaPowerTypeStruct power_type;
  struct nablaMainStruct *main;
  struct nablaVariableStruct *next;
}nablaVariable;


// ****************************************************************************
// * Nabla JOB struct
// ****************************************************************************
typedef struct nablaJobStruct{
  double struct_start;
  bool is_an_entry_point;
  bool is_a_function;
  bool has_to_be_unlinked;
  char *scope;
  char *region;
  char *item;
  char *item_set;
  int nb_in_item_set;
  char *return_type;
  char *name;
  char *name_utf8;
  char *xyz;
  char *direction;
  char at[NABLA_JOB_WHEN_MAX]; // Buffer de construction
  int when_index;
  double when_sign;
  int when_depth;
  double whens[NABLA_JOB_WHEN_MAX];
  char *where;
  astNode *jobNode;
  astNode *returnTypeNode;
  astNode *stdParamsNode;
  astNode *nblParamsNode;
  astNode *ifAfterAt;
  nablaVariable *used_variables;
  nablaOption *used_options;
  nablaVariable *called_variables;
  nablaVariable *variables_to_gather_scatter;
  char forall_item;
  bool reduction;
  char *reduction_name;
  int swirl_index;
  char enum_enum;
  bool exists;
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
    int iGather; // Indice de la variable qui est à gatherer
    int iScatter; // Indice de la variable qui est à scatterer
    // Bool pour savoir si l'on est en train de travailler
    // pour 'return'er depuis un argument du job courant
    // A voir si c'est encore utile
    bool returnFromArgument;
  } parse;
  struct nablaEntityStruct *entity;
  struct nablaJobStruct *next;
}nablaJob;


// ****************************************************************************
// * Nabla SWIRL & HLT struct
// ****************************************************************************
typedef struct nablaHLTStruct{
  int depth;
  double at[NABLA_JOB_WHEN_MAX];
}nablaHLT;
int nHLTCompare(const nablaHLT *,const nablaHLT *);

typedef struct nablaSwirlStruct{
  nablaHLT *at;
  nablaJob *jobs;
  struct nablaSwirlStruct *next;
} nablaSwirl;


// ****************************************************************************
// * Nabla ENTITY struct
// ****************************************************************************
typedef struct nablaEntityStruct{
  FILE *hdr, *src;
  const char *name;
  const char *name_upcase;
  nablaJob *jobs;
  enum_library libraries;
  struct nablaMainStruct *main; 
  struct nablaEntityStruct *next;
} nablaEntity;


// ****************************************************************************
// Nabla MAIN struct
// ****************************************************************************
typedef struct nablaMainStruct{
  astNode *root;
  FILE *main, *cfg, *axl, *dot;
  const char *name;
  char *tmpVarKinds;
  nablaVariable *variables;
  nablaType *types;
  nablaOption *options;
  nablaEntity *entity;
  nablaSwirl *swirls;
  NABLA_BACKEND backend;
  BACKEND_OPTION option;
  BACKEND_PARALLELISM parallelism;
  BACKEND_COMPILER compiler;
  char *interface_name; // Arcane specific
  char *specific_path; // Arcane specific
  char *service_name;   // Arcane specific
  int optionDumpTree;
  int HLT_depth;
  struct hookStruct *hook;
  struct callStruct *call;
} nablaMain;


// ****************************************************************************
// * HOOKs DEFINEs
// ****************************************************************************
#define HOOK(h,f) if (nabla->hook->h->f) nabla->hook->h->f(nabla)
#define cCALL(nabla,struct,fct)\
  (nabla->call->struct!=NULL)?\
  (nabla->call->struct->fct!=NULL)?\
   nabla->call->struct->fct():"":""

#define cHOOK(nabla,struct,fct)\
  (nabla->hook->struct!=NULL)?\
  (nabla->hook->struct->fct!=NULL)?\
   nabla->hook->struct->fct():"":""

#define cHOOKn(nabla,struct,fct)\
  (nabla->hook->struct!=NULL)?\
  (nabla->hook->struct->fct!=NULL)?\
  nabla->hook->struct->fct(nabla):"":""

#define cHOOKi(nabla,struct,fct,i)\
  (nabla->hook->struct!=NULL)?\
  (nabla->hook->struct->fct!=NULL)?\
  nabla->hook->struct->fct(i):"":""

#define cHOOKj(nabla,struct,fct,job)\
  (nabla->hook->struct!=NULL)?\
  (nabla->hook->struct->fct!=NULL)?\
   nabla->hook->struct->fct(job):"":""

// ****************************************************************************
// * Forward declaration
// ****************************************************************************
void middleGlobals(nablaMain*);
 
// nMiddleLibraries.c
bool isWithLibrary(nablaMain*,enum_library);
void nMiddleLibraries(astNode*,nablaEntity*);

// nMiddleEntities.c
void nMiddleEntityFree(nablaEntity*);
nablaEntity *nMiddleEntityNew(nablaMain*);
nablaEntity *nMiddleEntityAddEntity(nablaMain*,nablaEntity*);

// nMiddleJobs.c
void nMiddleScanForNablaJobParameter(astNode*,int,nablaMain*);
void nMiddleScanForNablaJobAtConstant(astNode*,nablaMain*);
char nMiddleScanForNablaJobForallItem(astNode*);
void nMiddleScanForIfAfterAt(astNode*,nablaJob*,nablaMain*);
void nMiddleDumpIfAfterAt(astNode*,nablaMain*,bool);
int nMiddleDumpParameterTypeList(nablaMain*,FILE*,astNode*);
nablaJob *nMiddleJobNew(nablaEntity*);
nablaJob *nMiddleJobAdd(nablaEntity*,nablaJob*);
nablaJob *nMiddleJobLast(nablaJob*);
nablaJob *nMiddleJobFind(nablaJob*,const char*);
void nMiddleJobParse(astNode*,nablaJob*);
void nMiddleJobFill(nablaMain*,nablaJob*,astNode*,const char*);
void nMiddleJobFree(nablaMain*);

// nMiddleVariables.c
void nMiddleVariableFree(nablaVariable*);
nablaVariable *nMiddleVariableNew(nablaMain*);
nablaVariable *nMiddleVariableAdd(nablaMain*,nablaVariable*);
nablaVariable *nMiddleVariableLast(nablaVariable*);
nablaVariable *nMiddleVariableFind(nablaVariable*,const char*);
nablaVariable *nMiddleVariableFindWithSameJobItem(nablaMain*,nablaJob*,
                                                  nablaVariable*,const char*);
what_to_do_with_the_postfix_expressions nMiddleVariables(nablaMain*,
                                                         astNode*,
                                                         const char,
                                                         char );
int nMiddleVariableGmpRank(nablaVariable*);
char *nMiddleVariableGmpNameRank(nablaVariable*,int);
bool nMiddleVariableGmpDumpRank(nablaVariable*,int);
int nMiddleVariableGmpDumpNumber(nablaVariable*);
void dfsEnumMax(nablaMain*,nablaJob*,astNode*);
void dfsExit(nablaMain*,nablaJob*,astNode*);
void dfsVariables(nablaMain*,nablaJob*,astNode*,bool);
void dfsVariablesDump(nablaMain*,nablaJob*,astNode*);
bool dfsUsedInThisForall(nablaMain*,nablaJob*,astNode*,const char*);

// nMiddleType
nablaType *nMiddleTypeNew(void);
nablaType *nMiddleTypeLast(nablaType*);
nablaType *nMiddleTypeAdd(nablaType*,nablaType*);
nablaType *nMiddleTypeFindName(nablaType*,char*);

// nMiddleOptions.c
void nMiddleOptionFree(nablaOption*);
nablaOption *nMiddleOptionNew(nablaMain*);
nablaOption *nMiddleOptionLast(nablaOption*);
nablaOption *nMiddleOptionAdd(nablaMain*,nablaOption*);
nablaOption *nMiddleOptionFindName(nablaOption*,const char*);
void nMiddleOptions(astNode*,int,nablaMain*);
nablaOption *nMiddleTurnTokenToOption(astNode*,nablaMain*);

// nMiddleHeader.c
NABLA_STATUS nMiddleInclude(nablaMain*,const char*);
NABLA_STATUS nMiddleDefines(nablaMain*,const nWhatWith*);
NABLA_STATUS nMiddleTypedefs(nablaMain*,const nWhatWith*);
NABLA_STATUS nMiddleForwards(nablaMain*,const char**);

// nMiddleGrammar.c
void nMiddleGrammar(astNode*,nablaMain*);
void nMiddleInsertSpace(nablaMain*,astNode*);

// nMiddle
NABLA_STATUS nMiddleSwitch(astNode*,
                           const int,
                           const char*,
                           const NABLA_BACKEND,
                           const BACKEND_OPTION,
                           const BACKEND_PARALLELISM,
                           const BACKEND_COMPILER,
                           char*,char*,char*);

// nMiddlePrintf
int nprintf(const struct nablaMainStruct*,const char*,const char*,...);
int hprintf(const struct nablaMainStruct*,const char*,const char*,...);

// nMiddleItems
void nMiddleItems(astNode*,int,nablaMain*);

// nMiddleHLT: @ + When[s]
void nMiddleAtConstantParse(nablaJob*,astNode*,nablaMain*);
void nMiddleStoreWhen(nablaJob*,nablaMain*);
int nMiddleComparEntryPoints(const void*,const void*);
int nMiddleNumberOfEntryPoints(nablaMain*);
nablaJob* nMiddleEntryPointsSort(nablaMain*,int);

// nMiddleTimeTree.c
NABLA_STATUS nMiddleTimeTreeSave(nablaMain*,nablaJob*,int);

// nMiddleFunctions
void nMiddleFunctionDumpHeader(FILE*,astNode*);
void nMiddleFunctionFill(nablaMain*,nablaJob*,astNode*,const char*);

NABLA_STATUS animate(nablaMain*);

void nMiddleArgsAddExtra(nablaMain*,int*);
void nMiddleArgsAddGlobal(nablaMain*,nablaJob*,int*);
void nMiddleArgsDump(nablaMain*,astNode*,int*);
void nMiddleArgsDumpFromDFS(nablaMain*,nablaJob*);
void nMiddleParamsDumpFromDFS(nablaMain*,nablaJob*,int);
void nMiddleParamsAddExtra(nablaMain*,int*);
void nMiddleDfsForCalls(nablaMain*,nablaJob*,astNode*,const char*,astNode*);

void nMiddleFunctionDumpFwdDeclaration(nablaMain*,nablaJob*,astNode*,const char *);

#endif // _NABLA_MIDDLEND_H_
