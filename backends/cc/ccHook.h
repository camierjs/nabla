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
#ifndef _NABLA_CC_H_
#define _NABLA_CC_H_

// SIMD
char* ccStdIncludes(void);
char* ccStdBits(void);
extern nablaTypedef ccStdTypedef[];
extern nablaDefine ccStdDefines[];
extern char* ccStdForwards[];


char* ccStdGather(nablaJob *,nablaVariable*,enum_phase);


char* ccStdScatter(nablaVariable*);


char* ccStdPrevCell(void);


char* ccStdNextCell(void);


// Cilk+ parallel color
char *ccParallelCilkSync(void);
char *ccParallelCilkSpawn(void);
char *ccParallelCilkLoop(nablaMain *);
char *ccParallelCilkIncludes(void);

// OpenMP parallel color
char *ccParallelOpenMPSync(void);
char *ccParallelOpenMPSpawn(void);
char *ccParallelOpenMPLoop(nablaMain *);
char *ccParallelOpenMPIncludes(void);

// Void parallel color
char *ccParallelVoidSync(void);
char *ccParallelVoidSpawn(void);
char *ccParallelVoidLoop(nablaMain *);
char *ccParallelVoidIncludes(void);

// Pragmas: Ivdep, Align
char *ccPragmaIccIvdep(void);
char *ccPragmaGccIvdep(void);
char *ccPragmaIccAlign(void);
char *ccPragmaGccAlign(void);

NABLA_STATUS ccMainPrefix(nablaMain*);
NABLA_STATUS ccMainPreInit(nablaMain*);
NABLA_STATUS ccMainVarInitKernel(nablaMain*);
NABLA_STATUS ccMainVarInitCall(nablaMain*);
NABLA_STATUS ccMainPostInit(nablaMain*);
NABLA_STATUS ccMain(nablaMain*);
NABLA_STATUS ccMainPostfix(nablaMain*);

void ccInlines(nablaMain*);
void ccDefineEnumerates(nablaMain*);
void ccVariablesPrefix(nablaMain*);
void ccVariablesPostfix(nablaMain*);

void ccMesh(nablaMain*);
void ccMainMeshPrefix(nablaMain*);
void ccMainMeshPostfix(nablaMain*);

void ccHookFunctionName(nablaMain*);
void ccHookFunction(nablaMain*, astNode*);
void ccHookJob(nablaMain*, astNode*);
void ccHookLibraries(astNode*, nablaEntity*);
char* ccHookPrefixEnumerate(nablaJob*);
char* ccHookDumpEnumerateXYZ(nablaJob*);
char* ccHookDumpEnumerate(nablaJob*);
char* ccHookPostfixEnumerate(nablaJob*);
char* ccHookItem(nablaJob*,const char, const char, char);
void ccHookSwitchToken(astNode*, nablaJob*);
nablaVariable *ccHookTurnTokenToVariable(astNode*,nablaMain*,nablaJob*);
void ccHookSystem(astNode*,nablaMain*, const char cnf, char enum_enum);
void ccHookAddExtraParameters(nablaMain*, nablaJob*, int *numParams);
void ccHookDumpNablaParameterList(nablaMain*, nablaJob*, astNode *n, int *numParams);
void ccHookTurnBracketsToParentheses(nablaMain*, nablaJob*, nablaVariable *var, char cnfg);
void ccHookJobDiffractStatement(nablaMain*, nablaJob*, astNode **n);

// Pour dumper les arguments necessaire dans le main
void ccDumpNablaArgumentList(nablaMain*, astNode *n, int *numParams);
void ccDumpNablaDebugFunctionFromOutArguments(nablaMain*, astNode *n,bool);
void ccAddExtraArguments(nablaMain*, nablaJob*, int *numParams);
void ccAddNablaVariableList(nablaMain*, astNode *n, nablaVariable **variables);
void ccAddExtraConnectivitiesParameters(nablaMain*, int*);
void ccAddExtraConnectivitiesArguments(nablaMain*, int*);

NABLA_STATUS cc(nablaMain*, astNode*, const char*);

#endif // _NABLA_CC_H_
 
