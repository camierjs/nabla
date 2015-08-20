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

char* lambdaHookBits(void);
char* lambdaHookGather(nablaJob*,nablaVariable*,enum_phase);
char* lambdaHookScatter(nablaVariable*);
char* lambdaHookPrevCell(void);
char* lambdaHookNextCell(void);
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

NABLA_STATUS lambdaMainPrefix(nablaMain*);
NABLA_STATUS lambdaMainPreInit(nablaMain*);
NABLA_STATUS lambdaMainVarInitKernel(nablaMain*);
NABLA_STATUS lambdaMainVarInitCall(nablaMain*);
NABLA_STATUS lambdaMainPostInit(nablaMain*);
NABLA_STATUS lambdaMain(nablaMain*);
NABLA_STATUS lambdaMainPostfix(nablaMain*);

void lambdaInlines(nablaMain*);
// static void lambdaDefineEnumerates(nablaMain*);
void lambdaVariablesPrefix(nablaMain*);
void lambdaVariablesPostfix(nablaMain*);

void lambdaMesh1D(nablaMain*);
void lambdaMesh3D(nablaMain*);
void lambdaMainMeshPrefix(nablaMain*);
void lambdaMainMeshPostfix(nablaMain*);

void lambdaHookIteration(struct nablaMainStruct*);
void lambdaHookExit(struct nablaMainStruct*);
void lambdaHookTime(struct nablaMainStruct*);
void lambdaHookFatal(struct nablaMainStruct*);
void lambdaHookAddCallNames(struct nablaMainStruct*,nablaJob*,astNode*);
bool lambdaHookPrimaryExpressionToReturn(nablaMain*,nablaJob*,astNode*);
char* lambdaHookEntryPointPrefix(struct nablaMainStruct*, nablaJob*);
void lambdaHookDfsForCalls(struct nablaMainStruct*,nablaJob*,astNode*,const char*,astNode*);
// static void lambdaHeaderPrefix(nablaMain*);
// static void lambdaHeaderIncludes(nablaMain*);
// static void lambdaHeaderDbg(nablaMain*);
// static void lambdaHeaderMth(nablaMain*);
// static void lambdaHeaderPostfix(nablaMain*);
// static void lambdaInclude(nablaMain*);

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
void lambdaHookJobDiffractStatement(nablaMain*,nablaJob*,astNode**);

// Pour dumper les arguments necessaire dans le main
void lambdaDumpNablaArgumentList(nablaMain*,astNode*,int*);
// static void lambdaDumpNablaDebugFunctionFromOutArguments(nablaMain*,astNode*,bool);
void lambdaAddExtraArguments(nablaMain*,nablaJob*,int*);
// static void lambdaAddExtraConnectivitiesArguments(nablaMain*,int*);
void lambdaHookAddExtraConnectivitiesParameters(nablaMain*,int*);

NABLA_STATUS nLambda(nablaMain*,astNode*,const char*);

#endif // _NABLA_LAMBDA_H_
 
