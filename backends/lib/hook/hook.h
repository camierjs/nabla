///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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
#ifndef _NABLA_LIB_HOOK_H_
#define _NABLA_LIB_HOOK_H_
//bool xHookSwitchForall(astNode *n, nablaJob *job);
//bool xHookSwitchAleph(astNode *n, nablaJob *job);

// forall
char* xHookForAllDump(nablaJob*);
char* xHookForAllItem(nablaJob*,const char,const char,char);
char* xHookForAllPostfix(nablaJob*);

// token
void xHookSwitchToken(node*, nablaJob*);
nablaVariable* xHookTurnTokenToVariable(node*,nablaMain*,nablaJob*);
void xHookTurnTokenToOption(nablaMain*,nablaOption*);
void xHookSystem(node*,nablaMain*,const char,char);
void xHookIteration(nablaMain*);
void xHookExit(nablaMain*,nablaJob*);
void xHookError(nablaMain*,nablaJob*);
void xHookTime(nablaMain*);
void xHookFatal(nablaMain*);
void xHookTurnBracketsToParentheses(nablaMain*,nablaJob*,nablaVariable*,char);
void xHookIsTest(nablaMain*, nablaJob*, node*, int);

// gram
void xHookReduction(nablaMain*,node*);
bool xHookDfsVariable(nablaMain*);

// call
void xHookAddCallNames(nablaMain*,nablaJob*,node*);
void xHookAddArguments(nablaMain*,nablaJob*);
char* xHookEntryPointPrefix(nablaMain*,nablaJob*);
void xHookDfsForCalls(nablaMain*,nablaJob*,node*,const char*,node*);

// xyz
char* xHookPrevCell(int);
char* xHookNextCell(int);
char* xHookSysPostfix(void);

// header
void xHookHeaderDump(nablaMain*);
void xHookHeaderDumpWithLibs(nablaMain*);
void xHookHeaderOpen(nablaMain*);
void xHookHeaderDefineEnumerates(nablaMain*);
void xHookHeaderPrefix(nablaMain*);
void xHookHeaderAlloc(nablaMain*);
void xHookHeaderIncludes(nablaMain*);
void xHookHeaderPostfix(nablaMain*);

// source
void xHookSourceOpen(nablaMain*);
void xHookSourceInclude(nablaMain*);
char* xHookSourceNamespace(nablaMain*);

// mesh
void xHookMeshPrefix(nablaMain*);
void xHookMesh1DConnectivity(nablaMain*);
void xHookMesh2DConnectivity(nablaMain*);
void xHookMesh3DConnectivity(nablaMain*,const char*);
void xHookMesh1D(nablaMain*);
void xHookMesh2D(nablaMain*);
void xHookMesh3D(nablaMain*);
void xHookMeshFreeConnectivity(nablaMain*);
void xHookMeshCore(nablaMain*);
void xHookMeshPostfix(nablaMain*);
void xHookMeshStruct(nablaMain*);

// vars
void xHookVariablesInit(nablaMain*);
void xHookVariablesPrefix(nablaMain*);
void xHookVariablesMalloc(nablaMain*);
void xHookVariablesFree(nablaMain*);
char* xHookVariablesODecl(nablaMain*);

// main
void xHookMainGLVisI2a(nablaMain*);
NABLA_STATUS xHookMainPrefix(nablaMain*);
NABLA_STATUS xHookMainPreInit(nablaMain*);
NABLA_STATUS xHookMainVarInitKernel(nablaMain*);
NABLA_STATUS xHookMainVarInitCall(nablaMain*);
NABLA_STATUS xHookMainHLT(nablaMain*);
NABLA_STATUS xHookMainPostInit(nablaMain*);
NABLA_STATUS xHookMainPostfix(nablaMain*);

#endif // _NABLA_LIB_HOOK_H_
 
