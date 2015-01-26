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
#ifndef _NABLA_CUDA_H_
#define _NABLA_CUDA_H_

char *nccCudaBits(void);
char* nccCudaGather(nablaJob*,nablaVariable*, enum_phase);
char* nccCudaScatter(nablaVariable*);
char* nccCudaPrevCell(void);
char* nccCudaNextCell(void);
char* nccCudaIncludes(void);

extern nablaTypedef cudaTypedef[];
extern nablaDefine cudaDefines[];
extern char* cudaForwards[];

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
 
