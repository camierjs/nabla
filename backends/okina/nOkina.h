/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccOkina.h      															  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2013.06.20																	  *
 * Updated  : 2013.06.20																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2013.06.20	camierjs	Creation															  *
 *****************************************************************************/
#ifndef _NABLA_OKINA_H_
#define _NABLA_OKINA_H_

// SIMD
char* okinaStdIncludes(void);
char* okinaSseIncludes(void);
char* okinaAvxIncludes(void);
char* okinaMicIncludes(void);

char* okinaStdBits(void);
char* okinaSseBits(void);
char* okinaAvxBits(void);
char* okinaMicBits(void);

extern nablaTypedef okinaStdTypedef[];
extern nablaTypedef okinaSseTypedef[];
extern nablaTypedef okinaAvxTypedef[];
extern nablaTypedef okinaMicTypedef[];

extern nablaDefine okinaStdDefines[];
extern nablaDefine okinaSseDefines[];
extern nablaDefine okinaAvxDefines[];
extern nablaDefine okinaMicDefines[];

extern char* okinaStdForwards[];
extern char* okinaSseForwards[];
extern char* okinaAvxForwards[];
extern char* okinaMicForwards[];

char* okinaStdGather(nablaJob *,nablaVariable*,enum_phase);
char* okinaSseGather(nablaJob *,nablaVariable*,enum_phase);
char* okinaAvxGather(nablaJob *,nablaVariable*,enum_phase);
char* okinaMicGather(nablaJob *,nablaVariable*,enum_phase);

char* okinaStdScatter(nablaVariable*);
char* okinaSseScatter(nablaVariable*);
char* okinaAvxScatter(nablaVariable*);
char* okinaMicScatter(nablaVariable*);

char* okinaStdPrevCell(void);
char* okinaSsePrevCell(void);
char* okinaAvxPrevCell(void);
char* okinaMicPrevCell(void);

char* okinaStdNextCell(void);
char* okinaSseNextCell(void);
char* okinaAvxNextCell(void);
char* okinaMicNextCell(void);


// Cilk+ parallel color
char *nccOkinaParallelCilkSync(void);
char *nccOkinaParallelCilkSpawn(void);
char *nccOkinaParallelCilkLoop(void);
char *nccOkinaParallelCilkIncludes(void);

// OpenMP parallel color
char *nccOkinaParallelOpenMPSync(void);
char *nccOkinaParallelOpenMPSpawn(void);
char *nccOkinaParallelOpenMPLoop(void);
char *nccOkinaParallelOpenMPIncludes(void);

// Void parallel color
char *nccOkinaParallelVoidSync(void);
char *nccOkinaParallelVoidSpawn(void);
char *nccOkinaParallelVoidLoop(void);
char *nccOkinaParallelVoidIncludes(void);

NABLA_STATUS nccOkinaMainPrefix(nablaMain*);
NABLA_STATUS nccOkinaMainPreInit(nablaMain*);
NABLA_STATUS nccOkinaMainVarInitKernel(nablaMain*);
NABLA_STATUS nccOkinaMainVarInitCall(nablaMain*);
NABLA_STATUS nccOkinaMainPostInit(nablaMain*);
NABLA_STATUS nccOkinaMain(nablaMain*);
NABLA_STATUS nccOkinaMainPostfix(nablaMain*);

void okinaInlines(nablaMain*);
void okinaDefineEnumerates(nablaMain*);
void okinaVariablesPrefix(nablaMain*);
void okinaVariablesPostfix(nablaMain*);

void okinaMesh(nablaMain*);
void nccOkinaMainMeshPrefix(nablaMain*);
void nccOkinaMainMeshPostfix(nablaMain*);

void okinaHookFunctionName(nablaMain*);
void okinaHookFunction(nablaMain*, astNode*);
void okinaHookJob(nablaMain*, astNode*);
void okinaHookLibraries(astNode*, nablaEntity*);
char* okinaHookPrefixEnumerate(nablaJob*);
char* okinaHookDumpEnumerateXYZ(nablaJob*);
char* okinaHookDumpEnumerate(nablaJob*);
char* okinaHookPostfixEnumerate(nablaJob*);
char* okinaHookItem(nablaJob*,const char, const char, char);
void okinaHookSwitchToken(astNode*, nablaJob*);
nablaVariable *okinaHookTurnTokenToVariable(astNode*,nablaMain*,nablaJob*);
void okinaHookSystem(astNode*,nablaMain*, const char cnf, char enum_enum);
void okinaHookAddExtraParameters(nablaMain*, nablaJob*, int *numParams);
void okinaHookDumpNablaParameterList(nablaMain*, nablaJob*, astNode *n, int *numParams);
void okinaHookTurnBracketsToParentheses(nablaMain*, nablaJob*, nablaVariable *var, char cnfg);
void okinaHookJobDiffractStatement(nablaMain*, nablaJob*, astNode **n);

// Pour dumper les arguments necessaire dans le main
void okinaDumpNablaArgumentList(nablaMain*, astNode *n, int *numParams);
void okinaDumpNablaDebugFunctionFromOutArguments(nablaMain*, astNode *n,bool);
void okinaAddExtraArguments(nablaMain*, nablaJob*, int *numParams);
void okinaAddNablaVariableList(nablaMain*, astNode *n, nablaVariable **variables);
void okinaAddExtraConnectivitiesParameters(nablaMain*, int*);
void okinaAddExtraConnectivitiesArguments(nablaMain*, int*);

NABLA_STATUS nccOkina(nablaMain*, astNode*, const char*);

#endif // _NABLA_OKINA_H_
 
