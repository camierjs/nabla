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
#ifndef _NABLA_H
#define _NABLA_H 

#include <math.h>
#include <stdio.h>
#include <iso646.h>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <stdbool.h>
int fileno(FILE *stream);
char *strdup(const char *s);
int mkstemp(char *template);
void nablaError(const char *format, ...);


// ****************************************************************************
// * Typedefs
// ****************************************************************************
typedef enum {
  NABLA_OK    = 0,
  NABLA_ERROR = ~NABLA_OK
} NABLA_STATUS;


// ****************************************************************************
// * Possible Backend Types: Arcane, Cuda or Okina
// ****************************************************************************
typedef enum {
  BACKEND_VOID   = 0,
  BACKEND_ARCANE = 1<<(0),
  BACKEND_CUDA   = 1<<(1),
  BACKEND_OKINA  = 1<<(2),
  BACKEND_CC     = 1<<(3),
} BACKEND_SWITCH;


// ****************************************************************************
// * Possible Variations:
// *  - Arcane: Module|Service,
// *  - Okina: std, sse, avx, avx2, mic and omp, cilk
// *  - TILING, SOA, AOS are not yet or will be used (to be cleaned-up!)
// ****************************************************************************
typedef enum { 
  BACKEND_COLOR_VOID           = 0,
  BACKEND_COLOR_ARCANE_ALONE   = 1<<(3),
  BACKEND_COLOR_ARCANE_MODULE  = 1<<(4),
  BACKEND_COLOR_ARCANE_SERVICE = 1<<(5),
  BACKEND_COLOR_OKINA_TILING   = 1<<(6),
  BACKEND_COLOR_OKINA_STD      = 1<<(7),
  BACKEND_COLOR_OKINA_SSE      = 1<<(8),
  BACKEND_COLOR_OKINA_AVX      = 1<<(9),
  BACKEND_COLOR_OKINA_AVX2     = 1<<(10),
  BACKEND_COLOR_OKINA_MIC      = 1<<(11),
  BACKEND_COLOR_OKINA_SEQ      = 1<<(12),
  BACKEND_COLOR_OpenMP         = 1<<(13),
  BACKEND_COLOR_CILK           = 1<<(14),
//  BACKEND_COLOR_OKINA_SOA      = 1<<(15),
//  BACKEND_COLOR_OKINA_AOS      = 1<<(16),
  BACKEND_COLOR_GCC            = 1<<(17),
  BACKEND_COLOR_ICC            = 1<<(18)
} BACKEND_COLORS;


// Enumération des phases possibles lors des génération des gather/scatter
typedef enum{
  enum_phase_declaration=0,
  enum_phase_function_call
} enum_phase;


// Enumération des actions à faire selon les cas de postfixs
typedef enum{
  postfixed_not_a_nabla_variable=0,
  postfixed_nabla_system_keyword=1,
  postfixed_nabla_variable_with_unknown=2,
  postfixed_nabla_variable_with_item=2
} what_to_do_with_the_postfix_expressions;


// ****************************************************************************
// * NABLA ENTRY_POINT_* DEFINES
// * __builtin_nanf ("")
// * __builtin_inff()
// * IEEE positive infinity (-HUGE_VAL is negative infinity).
// ****************************************************************************
#define ENTRY_POINT_build         (- (double)(__builtin_nanf("")))
#define ENTRY_POINT_init          (- (double)(__builtin_inff()))
#define ENTRY_POINT_start_init    (- (double)(0.0))
#define ENTRY_POINT_continue_init (- (double)(0.0))
#define ENTRY_POINT_compute_loop  (+ (double)(0.0))
#define ENTRY_POINT_exit          (+ (double)(__builtin_inff()))


// ****************************************************************************
// * NABLA STR & HARDCODED DEFINES
// ****************************************************************************
#define NABLA_LICENSE_HEADER 3360
#define NABLA_MAX_FILE_NAME 8192
#define NABLA_HARDCODED_VARIABLE_DIM_1_DEPTH 8

#include "frontend/nablaDebug.h"
#include "frontend/nablaAst.h"

#include "middlend/nablaMiddlend.h"

#include "backends/arcane/nccArcane.h"
#include "backends/cuda/nccCuda.h"
#include "backends/okina/nOkina.h"

#include "frontend/nablaTools.h"

#endif // _NABLA_H_
