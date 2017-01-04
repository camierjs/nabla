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
#ifndef _NABLA_OKINA_CALL_H_
#define _NABLA_OKINA_CALL_H_

// simd/[std|sse|avx|mic]
extern const char* nOkinaStdForwards[];
extern const char* nOkinaSseForwards[];
extern const char* nOkinaAvxForwards[];
extern const char* nOkina512Forwards[];
extern const char* nOkinaMicForwards[];

extern const nWhatWith nOkinaStdDefines[];
extern const nWhatWith nOkinaSseDefines[];
extern const nWhatWith nOkinaAvxDefines[];
extern const nWhatWith nOkina512Defines[];
extern const nWhatWith nOkinaMicDefines[];

extern const nWhatWith nOkinaStdTypedef[];
extern const nWhatWith nOkinaSseTypedef[];
extern const nWhatWith nOkinaAvxTypedef[];
extern const nWhatWith nOkina512Typedef[];
extern const nWhatWith nOkinaMicTypedef[];

char* nOkinaStdUid(nablaMain*,nablaJob*);
char* nOkinaSseUid(nablaMain*,nablaJob*);
char* nOkinaAvxUid(nablaMain*,nablaJob*);
char* nOkina512Uid(nablaMain*,nablaJob*);
char* nOkinaMicUid(nablaMain*,nablaJob*);

char* nOkinaStdIncludes(void);
char* nOkinaSseIncludes(void);
char* nOkinaAvxIncludes(void);
char* nOkina512Includes(void);
char* nOkinaMicIncludes(void);

char* nOkinaStdBits(void);
char* nOkinaSseBits(void);
char* nOkinaAvxBits(void);
char* nOkina512Bits(void);
char* nOkinaMicBits(void);

char* nOkinaStdGather(nablaJob*,nablaVariable*);
char* nOkinaSseGather(nablaJob*,nablaVariable*);
char* nOkinaAvxGather(nablaJob*,nablaVariable*);
char* nOkina512Gather(nablaJob*,nablaVariable*);
char* nOkinaMicGather(nablaJob*,nablaVariable*);

char* nOkinaStdScatter(nablaJob*,nablaVariable*);
char* nOkinaSseScatter(nablaJob*,nablaVariable*);
char* nOkinaAvxScatter(nablaJob*,nablaVariable*);
char* nOkina512Scatter(nablaJob*,nablaVariable*);
char* nOkinaMicScatter(nablaJob*,nablaVariable*);

char* nOkinaStdPrevCell(int);
char* nOkinaSsePrevCell(int);
char* nOkinaAvxPrevCell(int);
char* nOkina512PrevCell(int);
char* nOkinaMicPrevCell(int);

char* nOkinaStdNextCell(int);
char* nOkinaSseNextCell(int);
char* nOkinaAvxNextCell(int);
char* nOkina512NextCell(int);
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

// hooks/nOkinaHookGather
char* gather(astNode*,nablaJob*);

// hooks/nOkinaHookScatter
char* scatter(nablaJob*);

#endif // _NABLA_OKINA_CALL_H_
