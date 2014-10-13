/*****************************************************************************\
* File       : nabla.h																			*
* Author     : CamierJS 				                                          *
* Created    : 05.01.2010                                                     *
* Last update: 12.09.2013                                                     *
*******************************************************************************
*******************************************************************************
* Date	Author	Description												         		*
* 100105	camierjs	Initial version											         	*
* 120822	camierjs	Switch to arcane|XeonPhy|Cuda version				         	*
\*****************************************************************************/
#ifndef _NABLA_H
#define _NABLA_H 

#include <math.h>
#include <error.h>
#include <stdio.h>
#include <iso646.h>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <stdbool.h>
 

/*****************************************************************************
 * typedefs
 *****************************************************************************/
typedef enum {
  NABLA_OK=0,
  NABLA_ERROR=~NABLA_OK
} NABLA_STATUS;


// Type de backend possible: Arcane, Cuda ou Okina
typedef enum {
  BACKEND_VOID   = 0,
  BACKEND_ARCANE = 1<<(0),
  BACKEND_CUDA   = 1<<(1),
  BACKEND_OKINA  = 1<<(2),
} BACKEND_SWITCH;


// Type de variantes possibles:
//   - Arcane: Module|Service,
//   - Okina: Cartesian|Lagrangian
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
  BACKEND_COLOR_OKINA_OpenMP   = 1<<(13),
  BACKEND_COLOR_OKINA_CILK     = 1<<(14),
  BACKEND_COLOR_OKINA_SOA      = 1<<(15),
  BACKEND_COLOR_OKINA_AOS      = 1<<(16)
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
