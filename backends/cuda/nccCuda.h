// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
#ifndef _NABLA_CUDA_H_
#define _NABLA_CUDA_H_

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

NABLA_STATUS nccCuda(nablaMain*, astNode*, const char*);

#endif // _NABLA_CUDA_H_
 
