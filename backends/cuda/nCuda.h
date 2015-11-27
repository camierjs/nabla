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

void cudaHookSourceOpen(nablaMain *nabla);
void cudaHookSourceInclude(nablaMain *nabla);

char *cudaHookBits(void);
char* cudaHookGather(nablaJob*,nablaVariable*,enum_phase);
char* cudaHookScatter(nablaVariable*);

char* cudaHookSysPrefix(void);
char* cudaHookSysPostfix(void);
char* cudaHookPrevCell(void);
char* cudaHookNextCell(void);

char* cudaHookIncludes(void);

char* cudaHookFilterGather(nablaJob*);
char* cudaHookFilterScatter(nablaJob*);

void cudaHookReduction(struct nablaMainStruct*,astNode*);
void cudaHookAddArguments(struct nablaMainStruct*,nablaJob*);
//void lambdaHookReturnFromArgument(nablaMain*,nablaJob*);
void cudaHookTurnTokenToOption(struct nablaMainStruct*,nablaOption*);

char *cudaHookPragmaGccIvdep(void);
char *cudaHookPragmaGccAlign(void);

// Hooks: Header
void cudaHookHeaderDump(nablaMain*);
void cudaHookHeaderOpen(nablaMain*);
void cudaHookHeaderEnumerates(nablaMain*);
void cudaHookHeaderPrefix(nablaMain*);
void cudaHookHeaderPostfix(nablaMain*);
void cudaHookHeaderIncludes(nablaMain*);

// Dump into Header & Source
//void nLambdaDumpSource(nablaMain*);
void cudaHeaderTypes(nablaMain*);
void cudaHeaderReal3(nablaMain*);
void cudaHeaderExtra(nablaMain*);
void cudaHeaderMesh(nablaMain*);
void cudaHeaderHandleErrors(nablaMain*);
void cudaHeaderDebug(nablaMain*);
void cudaHeaderItems(nablaMain*);
//void nLambdaDumpMesh(nablaMain*);

NABLA_STATUS cudaHookMainPrefix(nablaMain*);
NABLA_STATUS cudaHookMainPreInit(nablaMain*);
NABLA_STATUS cudaHookMainVarInitKernel(nablaMain*);
NABLA_STATUS cudaHookMainVarInitCall(nablaMain*);
NABLA_STATUS cudaHookMainCore(nablaMain*);
NABLA_STATUS cudaHookMainPostInit(nablaMain*);
NABLA_STATUS cudaHookMainPostfix(nablaMain*);

void cudaHookVariablesInit(nablaMain*);
void cudaHookVariablesPrefix(nablaMain*);
void cudaHookVariablesPostfix(nablaMain*);
void cudaHookVariablesMalloc(nablaMain*);
void cudaHookVariablesFree(nablaMain*);

void cudaHookMeshPrefix(nablaMain*);
void cudaHookMeshCore(nablaMain*);
void cudaHookMeshPostfix(nablaMain*);
void cudaHookMeshConnectivity(nablaMain*);

void cudaHookIteration(struct nablaMainStruct*);
void cudaHookExit(struct nablaMainStruct*,nablaJob*);
void cudaHookTime(struct nablaMainStruct*);
void cudaHookFatal(struct nablaMainStruct*);
void cudaHookAddCallNames(struct nablaMainStruct*,nablaJob*,astNode*);
//bool lambdaHookPrimaryExpressionToReturn(nablaMain*,nablaJob*,astNode*);
char* cudaHookEntryPointPrefix(struct nablaMainStruct*,nablaJob*);
void cudaHookDfsForCalls(struct nablaMainStruct*,nablaJob*,astNode*,const char*,astNode*);

void cudaHookFunctionName(nablaMain*);
void cudaHookFunction(nablaMain*,astNode*);
void cudaHookJob(nablaMain*,astNode*);
void cudaHookLibraries(astNode*,nablaEntity*);

char* cudaHookForAllPrefix(nablaJob*);
char* cudaHookForAllDumpXYZ(nablaJob*);
char* cudaHookForAllDump(nablaJob*);
char* cudaHookForAllPostfix(nablaJob*);
char* cudaHookForAllItem(nablaJob*,const char,const char,char);

char* cudaHookTokenPrefix(nablaMain*);
char* cudaHookTokenPostfix(nablaMain*);

void cudaHookSwitchToken(astNode*,nablaJob*);
nablaVariable *cudaHookTurnTokenToVariable(astNode*,nablaMain*,nablaJob*);
void cudaHookSystem(astNode*,nablaMain*,const char,char);
void cudaHookAddExtraParameters(nablaMain*, nablaJob*, int*);
void cudaHookDumpNablaParameterList(nablaMain*,nablaJob*,astNode*,int *);
void cudaHookTurnBracketsToParentheses(nablaMain*,nablaJob*,nablaVariable*,char);

//void nLambdaHookDumpNablaArgumentList(nablaMain*,astNode*,int*);
//void nLambdaHookAddExtraArguments(nablaMain*,nablaJob*,int*);

void cudaHookJobDiffractStatement(nablaMain*,nablaJob*,astNode**);

void cudaHookIsTest(nablaMain*,nablaJob*,astNode*,int);


/*
void cudaInlines(nablaMain*);
void cudaDefineEnumerates(nablaMain*);
void cudaVariablesPrefix(nablaMain*);
void cudaVariablesPostfix(nablaMain*);

void cudaMesh(nablaMain*);
void cudaMeshConnectivity(nablaMain*);
void nccCudaMainMeshConnectivity(nablaMain*);
void nccCudaMainMeshPrefix(nablaMain*);
void nccCudaMainMeshPostfix(nablaMain*);

// Pour dumper les arguments necessaire dans le main
void cudaDumpNablaArgumentList(nablaMain*,astNode*,int*);
void cudaAddExtraArguments(nablaMain*, nablaJob*,int*);
void cudaAddExtraConnectivitiesParameters(nablaMain*,int*);
void cudaAddExtraConnectivitiesArguments(nablaMain*,int*);
*/

void cudaDumpNablaDebugFunctionFromOutArguments(nablaMain*,astNode*,bool);

nHooks *cuda(nablaMain*, astNode*);

#endif // _NABLA_CUDA_H_
 
