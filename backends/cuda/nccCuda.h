/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccCuda.h      																  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2012.12.13																	  *
 * Updated  : 2012.12.13																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2012.12.13	camierjs	Creation															  *
 *****************************************************************************/
#ifndef _NCC_CUDA_H_
#define _NCC_CUDA_H_

char *nccCudaBits(void);
char* nccCudaGather(nablaJob*,nablaVariable*, enum_phase);
char* nccCudaScatter(nablaVariable*);
char* nccCudaPrevCell(void);
char* nccCudaNextCell(void);
char* nccCudaIncludes(void);

NABLA_STATUS nccCudaMainPrefix(nablaMain *);
NABLA_STATUS nccCudaMainPreInit(nablaMain *);
NABLA_STATUS nccCudaMainVarInitKernel(nablaMain *);
NABLA_STATUS nccCudaMainVarInitCall(nablaMain *);
NABLA_STATUS nccCudaMainPostInit(nablaMain *);
NABLA_STATUS nccCudaMain(nablaMain *);
NABLA_STATUS nccCudaMainPostfix(nablaMain *);

void cudaInlines(nablaMain*);
void cudaDefineEnumerates(nablaMain*);
void cudaVariablesPrefix(nablaMain*);
void cudaVariablesPostfix(nablaMain*);

void cudaMesh(nablaMain*);
void cudaMeshConnectivity(nablaMain*);
void nccCudaMainMeshConnectivity(nablaMain *);
void nccCudaMainMeshPrefix(nablaMain *);
void nccCudaMainMeshPostfix(nablaMain *);

void cudaHookFunctionName(nablaMain *arc);
void cudaHookFunction(nablaMain *nabla, astNode *n);
void cudaHookJob(nablaMain *nabla, astNode *n);
void cudaHookLibraries(astNode * n, nablaEntity*);
char* cudaHookPrefixEnumerate(nablaJob *job);
char* cudaHookDumpEnumerateXYZ(nablaJob *job);
char* cudaHookDumpEnumerate(nablaJob *job);
char* cudaHookPostfixEnumerate(nablaJob *job);
char* cudaHookItem(nablaJob*, const char, const char, char);
void cudaHookSwitchToken(astNode *, nablaJob*);
nablaVariable *cudaHookTurnTokenToVariable(astNode*,nablaMain*,nablaJob*);
void cudaHookSystem(astNode * n,nablaMain*, const char cnf, char enum_enum);
void cudaHookAddExtraParameters(nablaMain*, nablaJob *job, int *numParams);
void cudaHookDumpNablaParameterList(nablaMain*, nablaJob*, astNode*,int *);
void cudaHookTurnBracketsToParentheses(nablaMain*, nablaJob *job, nablaVariable *var, char cnfg);
void cudaHookJobDiffractStatement(nablaMain*, nablaJob *job, astNode **n);

// Pour dumper les arguments necessaire dans le main
void cudaDumpNablaArgumentList(nablaMain *nabla, astNode *n, int *numParams);
void cudaDumpNablaDebugFunctionFromOutArguments(nablaMain *nabla, astNode *n,bool);
void cudaAddExtraArguments(nablaMain *nabla, nablaJob *job, int *numParams);
void cudaAddNablaVariableList(nablaMain *nabla, astNode *n, nablaVariable **variables);
void cudaAddExtraConnectivitiesParameters(nablaMain *nabla, int *numParams);
void cudaAddExtraConnectivitiesArguments(nablaMain *nabla, int *numParams);

NABLA_STATUS nccCuda(nablaMain*, astNode*, const char*, const char*);

#endif // _NABLA_CUDA_H_
 
