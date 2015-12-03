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

// ****************************************************************************
// * CUDA DUMP 
// ****************************************************************************
void cuHeaderTypes(nablaMain*);
void cuHeaderExtra(nablaMain*);
void cuHeaderMeshs(nablaMain*);
void cuHeaderError(nablaMain*);
void cuHeaderDebug(nablaMain*);
void cuHeaderItems(nablaMain*);


// ****************************************************************************
// * CUDA HOOKS
// ****************************************************************************
void cuHookSourceOpen(nablaMain *nabla);
void cuHookSourceInclude(nablaMain *nabla);

char *cuHookBits(void);
char* cuHookGather(nablaJob*,nablaVariable*,enum_phase);
char* cuHookScatter(nablaVariable*);

char* cuHookSysPrefix(void);
char* cuHookSysPostfix(void);
char* cuHookPrevCell(int);
char* cuHookNextCell(int);

char* cuHookIncludes(void);

char* cuHookFilterGather(nablaJob*);
char* cuHookFilterScatter(nablaJob*);

void cuHookReduction(struct nablaMainStruct*,astNode*);
void cuHookAddArguments(struct nablaMainStruct*,nablaJob*);
void cuHookTurnTokenToOption(struct nablaMainStruct*,nablaOption*);

char* cuHookPragmaGccIvdep(void);
char* cuHookPragmaGccAlign(void);

void cuHookHeaderDump(nablaMain*);
void cuHookHeaderOpen(nablaMain*);
void cuHookHeaderEnumerates(nablaMain*);
void cuHookHeaderPrefix(nablaMain*);
void cuHookHeaderPostfix(nablaMain*);
void cuHookHeaderIncludes(nablaMain*);

NABLA_STATUS cuHookMainPrefix(nablaMain*);
NABLA_STATUS cuHookMainPreInit(nablaMain*);
NABLA_STATUS cuHookMainVarInitKernel(nablaMain*);
NABLA_STATUS cuHookMainVarInitCall(nablaMain*);
NABLA_STATUS cuHookMainCore(nablaMain*);
NABLA_STATUS cuHookMainPostInit(nablaMain*);
NABLA_STATUS cuHookMainPostfix(nablaMain*);

void cuHookVariablesInit(nablaMain*);
void cuHookVariablesPrefix(nablaMain*);
void cuHookVariablesPostfix(nablaMain*);
void cuHookVariablesMalloc(nablaMain*);
void cuHookVariablesFree(nablaMain*);

void cuHookMeshPrefix(nablaMain*);
void cuHookMeshCore(nablaMain*);
void cuHookMeshPostfix(nablaMain*);
void cuHookMeshConnectivity(nablaMain*);

void cuHookIteration(struct nablaMainStruct*);
void cuHookExit(struct nablaMainStruct*,nablaJob*);
void cuHookTime(struct nablaMainStruct*);
void cuHookFatal(struct nablaMainStruct*);
void cuHookAddCallNames(struct nablaMainStruct*,nablaJob*,astNode*);
char* cuHookEntryPointPrefix(struct nablaMainStruct*,nablaJob*);
void cuHookDfsForCalls(struct nablaMainStruct*,nablaJob*,astNode*,const char*,astNode*);

void cuHookFunctionName(nablaMain*);
void cuHookFunction(nablaMain*,astNode*);
void cuHookJob(nablaMain*,astNode*);
void cuHookLibraries(astNode*,nablaEntity*);

char* cuHookForAllPrefix(nablaJob*);
char* cuHookForAllDumpXYZ(nablaJob*);
char* cuHookForAllDump(nablaJob*);
char* cuHookForAllPostfix(nablaJob*);
char* cuHookForAllItem(nablaJob*,const char,const char,char);

char* cuHookTokenPrefix(nablaMain*);
char* cuHookTokenPostfix(nablaMain*);

void cuHookSwitchToken(astNode*,nablaJob*);
nablaVariable *cuHookTurnTokenToVariable(astNode*,nablaMain*,nablaJob*);
void cuHookSystem(astNode*,nablaMain*,const char,char);
void cuHookAddExtraParameters(nablaMain*, nablaJob*, int*);
void cuHookDumpNablaParameterList(nablaMain*,nablaJob*,astNode*,int *);
void cuHookTurnBracketsToParentheses(nablaMain*,nablaJob*,nablaVariable*,char);

void cuHookJobDiffractStatement(nablaMain*,nablaJob*,astNode**);
void cuHookIsTest(nablaMain*,nablaJob*,astNode*,int);

void cuDumpNablaDebugFunctionFromOutArguments(nablaMain*,astNode*,bool);

hooks* cuda(nablaMain*);

#endif // _NABLA_CUDA_H_
 
