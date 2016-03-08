///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
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
#ifndef _NABLA_KOKKOS_HOOK_H_
#define _NABLA_KOKKOS_HOOK_H_

const hooks* kokkos(nablaMain*);
char* filterGather(astNode*,nablaJob*,GATHER_SCATTER_PHASE);
char* filterScatter(nablaJob*);


// ****************************************************************************
// * HOOKS
// ****************************************************************************
void kHookSourceOpen(nablaMain*);
void kHookSourceInclude(nablaMain*);
char* kHookSourceNamespace(nablaMain*);

void hookReduction(struct nablaMainStruct*,astNode*);
void hookTurnTokenToOption(struct nablaMainStruct*,nablaOption*);

char *hookPragmaGccAlign(void);

void hookMeshPrefix(nablaMain*);
void hookMeshCore(nablaMain*);
void hookMeshPostfix(nablaMain*);

void kHookIteration(struct nablaMainStruct*);

void hookFunctionName(nablaMain*);
void hookFunction(nablaMain*, astNode*);
void hookJob(nablaMain*, astNode*);
void hookLibraries(astNode*, nablaEntity*);

char* hookForAllPrefix(nablaJob*);
char* hookForAllDump(nablaJob*);
char* hookForAllPostfix(nablaJob*);
char* hookForAllItem(nablaJob*,const char, const char, char);

char* hookTokenPrefix(nablaMain*);
char* hookTokenPostfix(nablaMain*);

void hookSwitchToken(astNode*, nablaJob*);
void hookAddExtraParameters(nablaMain*,nablaJob*,int*);
void hookDumpNablaParameterList(nablaMain*,nablaJob*,astNode*,int*);
void hookAddExtraParametersDFS(nablaMain*,nablaJob*,int*);
void hookDumpNablaParameterListDFS(nablaMain*,nablaJob*,astNode*,int*);

void hookIsTest(nablaMain*,nablaJob*,astNode*,int);

#endif // _NABLA_KOKKOS_HOOK_H_
 
