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
#ifndef _NABLA_LAMBDA_H_
#define _NABLA_LAMBDA_H_

void nLambdaHookSourceOpen(nablaMain*);
void nLambdaHookSourceInclude(nablaMain*);

char* lambdaHookBits(void);
char* lambdaHookGather(nablaJob*,nablaVariable*,enum_phase);
char* lambdaHookScatter(nablaVariable*);

char* lambdaHookSysPrefix(void);
char* lambdaHookPrevCell(void);
char* lambdaHookNextCell(void);
char* lambdaHookSysPostfix(void);

char* lambdaHookIncludes(void);

char* lambdaHookFilterGather(nablaJob*);
char* lambdaHookFilterScatter(nablaJob*);

void lambdaHookReduction(struct nablaMainStruct*,astNode*);
void lambdaHookAddArguments(struct nablaMainStruct*,nablaJob*);
void lambdaHookReturnFromArgument(nablaMain*,nablaJob*);
void lambdaHookTurnTokenToOption(struct nablaMainStruct*,nablaOption*);

// Cilk+ parallel color
char *nLambdaParallelCilkSync(void);
char *nLambdaParallelCilkSpawn(void);
char *nLambdaParallelCilkLoop(nablaMain *);
char *nLambdaParallelCilkIncludes(void);

// OpenMP parallel color
char *nLambdaParallelOpenMPSync(void);
char *nLambdaParallelOpenMPSpawn(void);
char *nLambdaParallelOpenMPLoop(nablaMain *);
char *nLambdaParallelOpenMPIncludes(void);

// Void parallel color
char *nLambdaParallelVoidSync(void);
char *nLambdaParallelVoidSpawn(void);
char *nLambdaParallelVoidLoop(nablaMain *);
char *nLambdaParallelVoidIncludes(void);

// Pragmas: Ivdep, Align
char *lambdaHookPragmaIccIvdep(void);
char *lambdaHookPragmaGccIvdep(void);
char *lambdaHookPragmaIccAlign(void);
char *lambdaHookPragmaGccAlign(void);

// Hooks: Header
void nLambdaHookHeaderOpen(nablaMain *);
void nLambdaHookHeaderDefineEnumerates(nablaMain *);
void nLambdaHookHeaderPrefix(nablaMain *);
void nLambdaHookHeaderPostfix(nablaMain *);
void nLambdaHookHeaderIncludes(nablaMain *);

// Dump into Header & Source
void nLambdaHookHeaderDump(nablaMain *);
void nLambdaDumpSource(nablaMain*);
void nLambdaDumpHeaderTypes(nablaMain *);
void nLambdaDumpHeaderDebug(nablaMain *);
void nLambdaDumpHeaderMaths(nablaMain *);
void nLambdaDumpMesh(nablaMain*);

NABLA_STATUS nLambdaHookMainPrefix(nablaMain*);
NABLA_STATUS nLambdaHookMainPreInit(nablaMain*);
NABLA_STATUS nLambdaHookMainVarInitKernel(nablaMain*);
NABLA_STATUS nLambdaHookMainVarInitCall(nablaMain*);
NABLA_STATUS nLambdaHookMainPostInit(nablaMain*);
NABLA_STATUS nLambdaHookMain(nablaMain*);
NABLA_STATUS nLambdaHookMainPostfix(nablaMain*);

void nLambdaHookVariablesInit(nablaMain*);
void nLambdaHookVariablesPrefix(nablaMain*);
void nLambdaHookVariablesMalloc(nablaMain*);
void nLambdaHookVariablesFree(nablaMain*);

void nLambdaHookMeshPrefix(nablaMain*);
void nLambdaHookMeshCore(nablaMain*);
void nLambdaHookMeshPostfix(nablaMain*);


void lambdaHookIteration(struct nablaMainStruct*);
void lambdaHookExit(struct nablaMainStruct*);
void lambdaHookTime(struct nablaMainStruct*);
void lambdaHookFatal(struct nablaMainStruct*);
void lambdaHookAddCallNames(struct nablaMainStruct*,nablaJob*,astNode*);
bool lambdaHookPrimaryExpressionToReturn(nablaMain*,nablaJob*,astNode*);
char* lambdaHookEntryPointPrefix(struct nablaMainStruct*, nablaJob*);
void lambdaHookDfsForCalls(struct nablaMainStruct*,nablaJob*,astNode*,const char*,astNode*);

void lambdaHookFunctionName(nablaMain*);
void lambdaHookFunction(nablaMain*, astNode*);
void lambdaHookJob(nablaMain*, astNode*);
void lambdaHookLibraries(astNode*, nablaEntity*);
char* lambdaHookPrefixEnumerate(nablaJob*);
char* lambdaHookDumpEnumerateXYZ(nablaJob*);
char* lambdaHookDumpEnumerate(nablaJob*);
char* lambdaHookPostfixEnumerate(nablaJob*);
char* lambdaHookItem(nablaJob*,const char, const char, char);
void lambdaHookSwitchToken(astNode*, nablaJob*);
nablaVariable *lambdaHookTurnTokenToVariable(astNode*,nablaMain*,nablaJob*);
void lambdaHookSystem(astNode*,nablaMain*,const char,char);
void lambdaHookAddExtraParameters(nablaMain*,nablaJob*,int*);
void lambdaHookDumpNablaParameterList(nablaMain*,nablaJob*,astNode*,int*);
void lambdaHookTurnBracketsToParentheses(nablaMain*,nablaJob*,nablaVariable*,char);

// Pour dumper les arguments necessaire dans le main
void nLambdaHookDumpNablaArgumentList(nablaMain*,astNode*,int*);
void nLambdaHookAddExtraArguments(nablaMain*,nablaJob*,int*);

nHooks *nLambda(nablaMain*);

#endif // _NABLA_LAMBDA_H_
 
