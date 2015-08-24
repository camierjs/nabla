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

char *nCudaHookBits(void);
char* nCudaHookGather(nablaJob*,nablaVariable*,enum_phase);
char* nCudaHookScatter(nablaVariable*);
char* nCudaHookPrevCell(void);
char* nCudaHookNextCell(void);
char* nCudaHookIncludes(void);

extern nWhatWith nCudaHookTypedef[];
extern nWhatWith nCudaHookDefines[];
extern char* nCudaHookForwards[];

NABLA_STATUS nccCudaMainPrefix(nablaMain*);
NABLA_STATUS nccCudaMainPreInit(nablaMain*);
NABLA_STATUS nccCudaMainVarInitKernel(nablaMain*);
NABLA_STATUS nccCudaMainVarInitCall(nablaMain*);
NABLA_STATUS nccCudaMainPostInit(nablaMain*);
NABLA_STATUS nccCudaMain(nablaMain*);
NABLA_STATUS nccCudaMainPostfix(nablaMain*);

void nCudaInlines(nablaMain*);
void cudaDefineEnumerates(nablaMain*);
void cudaVariablesPrefix(nablaMain*);
void cudaVariablesPostfix(nablaMain*);

void cudaHeaderReal3(nablaMain*);
void cudaHeaderExtra(nablaMain*);
void cudaHeaderMesh(nablaMain*);
void cudaHeaderHandleErrors(nablaMain*);
void cudaHeaderDebug(nablaMain*);

void cudaMesh(nablaMain*);
void cudaMeshConnectivity(nablaMain*);
void nccCudaMainMeshConnectivity(nablaMain*);
void nccCudaMainMeshPrefix(nablaMain*);
void nccCudaMainMeshPostfix(nablaMain*);

void nCudaHookFunctionName(nablaMain*);
void nCudaHookFunction(nablaMain*,astNode*);
void nCudaHookJob(nablaMain*,astNode*);
void nCudaHookLibraries(astNode*,nablaEntity*);
char* nCudaHookPrefixEnumerate(nablaJob*);
char* nCudaHookDumpEnumerateXYZ(nablaJob*);
char* nCudaHookDumpEnumerate(nablaJob*);
char* nCudaHookPostfixEnumerate(nablaJob*);
char* nCudaHookItem(nablaJob*,const char,const char,char);
void nCudaHookSwitchToken(astNode*,nablaJob*);
nablaVariable *nCudaHookTurnTokenToVariable(astNode*,nablaMain*,nablaJob*);
void nCudaHookSystem(astNode*,nablaMain*,const char,char);
void nCudaHookAddExtraParameters(nablaMain*, nablaJob*, int*);
void nCudaHookDumpNablaParameterList(nablaMain*,nablaJob*,astNode*,int *);
void nCudaHookTurnBracketsToParentheses(nablaMain*,nablaJob*,nablaVariable*,char);
void nCudaHookJobDiffractStatement(nablaMain*,nablaJob*,astNode**);
void nCudaHookReduction(struct nablaMainStruct*,astNode *);

void nCudaHookIteration(struct nablaMainStruct*);
void nCudaHookExit(struct nablaMainStruct*);
void nCudaHookTime(struct nablaMainStruct*);
void nCudaHookFatal(struct nablaMainStruct*);
void nCudaHookAddCallNames(struct nablaMainStruct*,nablaJob*,astNode*);
void nCudaHookAddArguments(struct nablaMainStruct*,nablaJob*);
void nCudaHookTurnTokenToOption(struct nablaMainStruct*,nablaOption*);
char* nCudaHookEntryPointPrefix(struct nablaMainStruct*,nablaJob*);
void nCudaHookDfsForCalls(struct nablaMainStruct*,nablaJob*,astNode*,const char*,astNode*);

char *nCudaPragmaGccIvdep(void);
char *nCudaPragmaGccAlign(void);
char* cudaGather(nablaJob*);
char* cudaScatter(nablaJob*);


// Pour dumper les arguments necessaire dans le main
void cudaDumpNablaArgumentList(nablaMain*,astNode*,int*);
void cudaDumpNablaDebugFunctionFromOutArguments(nablaMain*,astNode*,bool);
void cudaAddExtraArguments(nablaMain*, nablaJob*,int*);
void cudaAddNablaVariableList(nablaMain*,astNode*,nablaVariable**);
void cudaAddExtraConnectivitiesParameters(nablaMain*,int*);
void cudaAddExtraConnectivitiesArguments(nablaMain*,int*);

NABLA_STATUS nccCuda(nablaMain*,astNode*,const char*);

#endif // _NABLA_CUDA_H_
 
