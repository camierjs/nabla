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
char *nccOkinaParallelCilkLoop(nablaMain *);
char *nccOkinaParallelCilkIncludes(void);

// OpenMP parallel color
char *nccOkinaParallelOpenMPSync(void);
char *nccOkinaParallelOpenMPSpawn(void);
char *nccOkinaParallelOpenMPLoop(nablaMain *);
char *nccOkinaParallelOpenMPIncludes(void);

// Void parallel color
char *nccOkinaParallelVoidSync(void);
char *nccOkinaParallelVoidSpawn(void);
char *nccOkinaParallelVoidLoop(nablaMain *);
char *nccOkinaParallelVoidIncludes(void);

// Pragmas: Ivdep, Align
char *nccOkinaPragmaIccIvdep(void);
char *nccOkinaPragmaGccIvdep(void);
char *nccOkinaPragmaIccAlign(void);
char *nccOkinaPragmaGccAlign(void);

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

void okinaMesh1D(nablaMain*);
void okinaMesh3D(nablaMain*);
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
 
