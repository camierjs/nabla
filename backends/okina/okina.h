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
#ifndef _NABLA_OKINA_H_
#define _NABLA_OKINA_H_

// simd/[std|sse|avx|mic]
extern const char* nOkinaStdForwards[];
extern const char* nOkinaSseForwards[];
extern const char* nOkinaAvxForwards[];
extern const char* nOkinaMicForwards[];

extern const nWhatWith nOkinaStdDefines[];
extern const nWhatWith nOkinaSseDefines[];
extern const nWhatWith nOkinaAvxDefines[];
extern const nWhatWith nOkinaMicDefines[];

extern const nWhatWith nOkinaStdTypedef[];
extern const nWhatWith nOkinaSseTypedef[];
extern const nWhatWith nOkinaAvxTypedef[];
extern const nWhatWith nOkinaMicTypedef[];

char* nOkinaStdIncludes(void);
char* nOkinaSseIncludes(void);
char* nOkinaAvxIncludes(void);
char* nOkinaMicIncludes(void);

char* nOkinaStdBits(void);
char* nOkinaSseBits(void);
char* nOkinaAvxBits(void);
char* nOkinaMicBits(void);

char* nOkinaStdGather(nablaJob*,nablaVariable*,enum_phase);
char* nOkinaSseGather(nablaJob*,nablaVariable*,enum_phase);
char* nOkinaAvxGather(nablaJob*,nablaVariable*,enum_phase);
char* nOkinaMicGather(nablaJob*,nablaVariable*,enum_phase);

char* nOkinaStdScatter(nablaVariable*);
char* nOkinaSseScatter(nablaVariable*);
char* nOkinaAvxScatter(nablaVariable*);
char* nOkinaMicScatter(nablaVariable*);

char* nOkinaHookSysPrefix(void);
char* nOkinaHookSysPostfix(void);

char* nOkinaStdPrevCell(int);
char* nOkinaSsePrevCell(int);
char* nOkinaAvxPrevCell(int);
char* nOkinaMicPrevCell(int);

char* nOkinaStdNextCell(int);
char* nOkinaSseNextCell(int);
char* nOkinaAvxNextCell(int);
char* nOkinaMicNextCell(int);

// Cilk+ parallel color
char *nOkinaParallelCilkSync(void);
char *nOkinaParallelCilkSpawn(void);
char *nOkinaParallelCilkLoop(nablaMain *);
char *nOkinaParallelCilkIncludes(void);

// OpenMP parallel color
char *nOkinaParallelOpenMPSync(void);
char *nOkinaParallelOpenMPSpawn(void);
char *nOkinaParallelOpenMPLoop(nablaMain *);
char *nOkinaParallelOpenMPIncludes(void);

// Void parallel color
char *nOkinaParallelVoidSync(void);
char *nOkinaParallelVoidSpawn(void);
char *nOkinaParallelVoidLoop(nablaMain *);
char *nOkinaParallelVoidIncludes(void);

// Pragmas: Ivdep, Align
char *nOkinaPragmaIccIvdep(void);
char *nOkinaPragmaGccIvdep(void);
char *nOkinaPragmaIccAlign(void);
char *nOkinaPragmaGccAlign(void);

// nOkinaInit
NABLA_STATUS nOkinaInitVariables(nablaMain*);
NABLA_STATUS nOkinaInitVariableDbg(nablaMain*);

// nOkinaMain
NABLA_STATUS nOkinaMainPrefix(nablaMain*);
NABLA_STATUS nOkinaMainPreInit(nablaMain*);
NABLA_STATUS nOkinaMainPostInit(nablaMain*);
NABLA_STATUS nOkinaMain(nablaMain*);
NABLA_STATUS nOkinaMainPostfix(nablaMain*);

// nOkinaEnum
void nOkinaEnumDefine(nablaMain*);

// nOkinaVariables
void okinaVariablesPrefix(nablaMain*);
void okinaVariablesPostfix(nablaMain*);

// nOkinaHeader
void nOkinaHeaderInclude(nablaMain *nabla);
void nOkinaHeaderPrefix(nablaMain *nabla);
void nOkinaHeaderIncludes(nablaMain *nabla);
void nOkinaHeaderSimd(nablaMain *nabla);
void nOkinaHeaderDbg(nablaMain *nabla);
void nOkinaHeaderMth(nablaMain *nabla);
void nOkinaHeaderPostfix(nablaMain *nabla);

// hooks/nOkinaHookVariables
nablaVariable *nOkinaHookVariablesTurnTokenToVariable(astNode*,nablaMain*,nablaJob*);
void nOkinaHookVariablesSystem(astNode*,nablaMain*, const char, char);
void nOkinaHookVariablesTurnBracketsToParentheses(nablaMain*, nablaJob*, nablaVariable*, char );

// hooks/nOkinaHooks
void nOkinaHookDfsForCalls(struct nablaMainStruct*,nablaJob*,astNode*,const char* namespace,astNode*);
char* nOkinaHookEntryPointPrefix(struct nablaMainStruct*,nablaJob*);
void nOkinaHookIteration(struct nablaMainStruct*);
void nOkinaHookExit(struct nablaMainStruct*, nablaJob*);
void nOkinaHookTime(struct nablaMainStruct*);
void nOkinaHookFatal(struct nablaMainStruct*);
void nOkinaHookAddCallNames(struct nablaMainStruct*,nablaJob*,astNode*);
void nOkinaHookAddArguments(struct nablaMainStruct*,nablaJob*);
void nOkinaHookTurnTokenToOption(struct nablaMainStruct*,nablaOption*);
void nOkinaHookLibraries(astNode *, nablaEntity*);
void nOkinaHookJob(nablaMain*, astNode*);
bool nOkinaHookPrimaryExpressionToReturn(nablaMain*, nablaJob*, astNode*);
void nOkinaHookReturnFromArgument(nablaMain*, nablaJob*);

// hooks/nOkinaHookReduction
void nOkinaHookReduction(struct nablaMainStruct*, astNode*);

// hooks/nOkinaHookItem
char* nOkinaHookItem(nablaJob*,const char, const char, char);

// hooks/nOkinaHookFunction
void nOkinaHookFunctionName(nablaMain*);
void nOkinaHookFunction(nablaMain*, astNode*);

// hooks/nOkinaHookDiffraction
void nOkinaHookDiffraction(nablaMain*, nablaJob*, astNode**);

// hooks/nOkinaHookEnumerate
char* nOkinaHookEnumeratePrefix(nablaJob*);
char* nOkinaHookEnumerateDumpXYZ(nablaJob*);
char* nOkinaHookEnumerateDump(nablaJob*);
char* nOkinaHookEnumeratePostfix(nablaJob*);

// hooks/nOkinaHookGather
char* nOkinaHookGather(nablaJob*);

// hooks/nOkinaHookScatter
char* nOkinaHookScatter(nablaJob*);

// hooks/nOkinaHookToken
void nOkinaHookTokenSwitch(astNode*, nablaJob*);

void okinaHookIsTest(nablaMain*,nablaJob*,astNode*,int);

// mesh/nOkinaMesh
void nOkinaMeshPrefix(nablaMain*);
void nOkinaMeshPostfix(nablaMain*);
void nOkinaMesh1D(nablaMain*);
void nOkinaMesh3D(nablaMain*);

// nOkinaArgs
void nOkinaArgsAddExtraConnectivities(nablaMain*, int*);
void nOkinaArgsExtra(nablaMain*, nablaJob*, int*);
void nOkinaArgsList(nablaMain *, astNode *, int *);
void nOkinaArgsDumpNablaDebugFunctionFromOut(nablaMain*, astNode*,bool);

// hooks/nOkinaHookParams
void nOkinaHookParamsAddExtra(nablaMain*, nablaJob*, int*);
void nOkinaHookParamsDumpList(nablaMain*, nablaJob*, astNode*, int*);

char* nOkinaHookTokenPrefix(nablaMain*);
char* nOkinaHookTokenPostfix(nablaMain*);

// nOkina
NABLA_STATUS nOkina(nablaMain*, astNode*, const char*);

#endif // _NABLA_OKINA_H_
 
