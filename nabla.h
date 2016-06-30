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


// ****************************************************************************
// * Typedefs
// ****************************************************************************
typedef enum {
  NABLA_OK    = 0,
  NABLA_ERROR = ~NABLA_OK
} NABLA_STATUS;

// ToUpperCase[IntegerString[Hash[OPTION_TIME _DOT _MMA, "CRC32"], 16]]

// ****************************************************************************
// * Possible Options on the command line
// ****************************************************************************
typedef enum {
  OPTION_HELP         = 0x9ba9fd76,
  OPTION_ORGOPT       = 0x3c0f6f4c,
  OPTION_VERSION      = 0x7be945c5,
  OPTION_LICENSE      = 0xa534354f,
  OPTION_TIME_DOT_STD = 0xcb7e90e4,
  OPTION_TIME_DOT_MMA = 0x91791b82
} NABLA_OPTION;


// ****************************************************************************
// * Possible Backend Types: Arcane, Cuda or Okina
// ****************************************************************************
typedef enum {
  BACKEND_CUDA        = 0xd721d0f8,
  BACKEND_RAJA        = 0x6a1447b7,
  BACKEND_OKINA       = 0x3fda1e56,
  BACKEND_ARCANE      = 0xcbbe711d,
  BACKEND_LAMBDA      = 0x1393b187,
  BACKEND_KOKKOS      = 0x5a83b2b0
//  BACKEND_LOCI      = 0x9FE3840F,
//  BACKEND_UINTAH    = 0x12B30ED1,
//  BACKEND_MMA       = 0xE949383D,
//  BACKEND_LIBRARY   = 0x1F29F1EC,
//  BACKEND_VHDL      = 0x149283B0
} NABLA_BACKEND;


// ****************************************************************************
// *  - ARCANE: Alone|Family|Module|Service,
// *  -  OKINA: std|sse|avx|avx2|mic
// ****************************************************************************
typedef enum { 
  BACKEND_OPTION_ARCANE_ALONE   = 0xF1B4A200,
  BACKEND_OPTION_ARCANE_FAMILY  = 0x82B7457A,
  BACKEND_OPTION_ARCANE_MODULE  = 0xC40CAD4C,
  BACKEND_OPTION_ARCANE_SERVICE = 0x2F0572BB,
  BACKEND_OPTION_OKINA_STD      = 0x316E93D5,
  BACKEND_OPTION_OKINA_SSE      = 0x902C2E9F,
  BACKEND_OPTION_OKINA_AVX      = 0x4B333B9E,
  BACKEND_OPTION_OKINA_AVX2     = 0x11319999,
  BACKEND_OPTION_OKINA_AVX512   = 0xBBB3E00F,
  BACKEND_OPTION_OKINA_MIC      = 0xA5C9371C
} BACKEND_OPTION;

typedef enum { 
  BACKEND_PARALLELISM_SEQ       = 0x9FC33629,
  BACKEND_PARALLELISM_OMP       = 0x8F76BB72,
  BACKEND_PARALLELISM_CILK      = 0xC7DD275F
} BACKEND_PARALLELISM;

typedef enum { 
  BACKEND_COMPILER_GCC          = 0x269B4156,
  BACKEND_COMPILER_ICC          = 0x71C5A3D3
} BACKEND_COMPILER;


// ****************************************************************************
// * Enumération des actions à faire selon les cas de postfixs
// ****************************************************************************
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
// * Forwards
// ****************************************************************************
int fileno(FILE *stream);
char *strdup(const char *s);
int mkstemp(char *template);
#define nablaError(...) nablaErrorVariadic(__FILE__,__LINE__,__VA_ARGS__)
void nablaErrorVariadic(const char *,const int,const char *,...);


// ****************************************************************************
// * Nabla Hard-coded Definitions that should be removed
// ****************************************************************************
#define NABLA_LICENSE_HEADER 3360
#define NABLA_JOB_WHEN_MAX 64
#define NABLA_MAX_FILE_NAME 8192
#define NABLA_JOB_WHEN_HLT_FACTOR 3
#define NABLA_HARDCODED_VARIABLE_DIM_1_DEPTH 8


// ****************************************************************************
// * Std ∇ includes
// ****************************************************************************
#include "frontend/frontend.h"
#include "middlend/middlend.h"

#include "backends/calls.h"
#include "backends/hooks.h"

#include "backends/arcane/arcane.h"
#include "backends/cuda/cuda.h"
#include "backends/kokkos/kokkos.h"
#include "backends/lambda/lambda.h"
#include "backends/okina/okina.h"
#include "backends/raja/raja.h"

#include "backends/lib/call/call.h"
#include "backends/lib/hook/hook.h"
#include "backends/lib/dump/dump.h"

#include "toolbox/toolbox.h"

#endif // _NABLA_H_
