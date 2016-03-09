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
#ifndef _NABLA_BACKENDS_CALLS_H_
#define _NABLA_BACKENDS_CALLS_H_

typedef struct callHeaderStruct{
  const char** forwards;
  const nWhatWith* defines;
  const nWhatWith* typedefs;
} callHeader;

// Structure des calls que l'on va utiliser afin de générer pour AVX ou MIC
typedef struct callSimdStruct{
  char* (*bits)(void);
  char* (*gather)(nablaJob*,nablaVariable*);
  char* (*scatter)(nablaVariable*);
  char* (*includes)(void);
} callSimd;


// Structure des calls de gestion du parallelisme
typedef struct callParallelStruct{
  char* (*sync)(void);
  char* (*spawn)(void);
  char* (*loop)(struct nablaMainStruct*);
  char* (*includes)(void);
} callParallel;


// ****************************************************************************
// * Backend CALLS
// ****************************************************************************
typedef struct callStruct{
  const callHeader *header;
  const callSimd *simd; 
  const callParallel *parallel;
} backendCalls;


#endif // _NABLA_BACKENDS_CALLS_H_
