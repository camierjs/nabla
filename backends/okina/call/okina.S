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
   //.section ".rodata"
	.globl ${ABI_UNDERSCORE}knDbg_h
	.globl ${ABI_UNDERSCORE}knMsh1D_c
	.globl ${ABI_UNDERSCORE}knMsh3D_c
	.globl ${ABI_UNDERSCORE}knMth_h
	        
	.globl ${ABI_UNDERSCORE}knStdReal_h
	.globl ${ABI_UNDERSCORE}knStdReal3_h
	.globl ${ABI_UNDERSCORE}knStdInteger_h
	.globl ${ABI_UNDERSCORE}knStdTernary_h
	.globl ${ABI_UNDERSCORE}knStdGather_h
	.globl ${ABI_UNDERSCORE}knStdScatter_h
	.globl ${ABI_UNDERSCORE}knStdOStream_h

	.globl ${ABI_UNDERSCORE}knSseReal_h
	.globl ${ABI_UNDERSCORE}knSseReal3_h
	.globl ${ABI_UNDERSCORE}knSseInteger_h
	.globl ${ABI_UNDERSCORE}knSseTernary_h
	.globl ${ABI_UNDERSCORE}knSseGather_h
	.globl ${ABI_UNDERSCORE}knSseScatter_h
	.globl ${ABI_UNDERSCORE}knSseOStream_h
   
   .globl ${ABI_UNDERSCORE}knAvxReal_h
   .globl ${ABI_UNDERSCORE}knAvxReal3_h
   .globl ${ABI_UNDERSCORE}knAvxInteger_h
   .globl ${ABI_UNDERSCORE}knAvxTernary_h
   .globl ${ABI_UNDERSCORE}knAvxGather_h
   .globl ${ABI_UNDERSCORE}knAvx2Gather_h
   .globl ${ABI_UNDERSCORE}knAvxScatter_h
   .globl ${ABI_UNDERSCORE}knAvxOStream_h
        
   .globl ${ABI_UNDERSCORE}knMicReal_h
   .globl ${ABI_UNDERSCORE}knMicReal3_h
   .globl ${ABI_UNDERSCORE}knMicInteger_h
   .globl ${ABI_UNDERSCORE}knMicTernary_h
   .globl ${ABI_UNDERSCORE}knMicGather_h
   .globl ${ABI_UNDERSCORE}knMicScatter_h
   .globl ${ABI_UNDERSCORE}knMicOStream_h
        
${ABI_UNDERSCORE}knDbg_h:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/debug/knDbg.h"
   .byte 0
${ABI_UNDERSCORE}knMsh1D_c: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/mesh/knMsh1D.c"
   .byte 0
${ABI_UNDERSCORE}knMsh3D_c: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/mesh/knMsh3D.c"
   .byte 0
${ABI_UNDERSCORE}knMth_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/maths/knMth.h"
   .byte 0


        
${ABI_UNDERSCORE}knStdReal_h:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/std/knStdReal.h"
   .byte 0
${ABI_UNDERSCORE}knStdReal3_h:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/std/knStdReal3.h"
   .byte 0
${ABI_UNDERSCORE}knStdInteger_h:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/std/knStdInteger.h"
   .byte 0
${ABI_UNDERSCORE}knStdTernary_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/std/knStdTernary.h"
   .byte 0
${ABI_UNDERSCORE}knStdGather_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/std/knStdGather.h"
   .byte 0
${ABI_UNDERSCORE}knStdScatter_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/std/knStdScatter.h"
   .byte 0
${ABI_UNDERSCORE}knStdOStream_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/std/knStdOStream.h"
   .byte 0


        
${ABI_UNDERSCORE}knSseReal_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/sse/knSseReal.h"
        .byte 0
${ABI_UNDERSCORE}knSseReal3_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/sse/knSseReal3.h"
        .byte 0
${ABI_UNDERSCORE}knSseInteger_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/sse/knSseInteger.h"
        .byte 0
${ABI_UNDERSCORE}knSseTernary_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/sse/knSseTernary.h"
        .byte 0
${ABI_UNDERSCORE}knSseGather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/sse/knSseGather.h"
        .byte 0
${ABI_UNDERSCORE}knSseScatter_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/sse/knSseScatter.h"
        .byte 0
${ABI_UNDERSCORE}knSseOStream_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/sse/knSseOStream.h"
        .byte 0


        
${ABI_UNDERSCORE}knAvxReal_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/avx/knAvxReal.h"
        .byte 0
${ABI_UNDERSCORE}knAvxReal3_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/avx/knAvxReal3.h"
        .byte 0
${ABI_UNDERSCORE}knAvxInteger_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/avx/knAvxInteger.h"
        .byte 0
${ABI_UNDERSCORE}knAvxTernary_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/avx/knAvxTernary.h"
        .byte 0
${ABI_UNDERSCORE}knAvxGather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/avx/knAvxGather.h"
        .byte 0
${ABI_UNDERSCORE}knAvx2Gather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/avx/knAvx2Gather.h"
        .byte 0
${ABI_UNDERSCORE}knAvxScatter_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/avx/knAvxScatter.h"
        .byte 0
${ABI_UNDERSCORE}knAvxOStream_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/avx/knAvxOStream.h"
        .byte 0


        
${ABI_UNDERSCORE}knMicReal_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/mic/knMicReal.h"
        .byte 0
${ABI_UNDERSCORE}knMicReal3_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/mic/knMicReal3.h"
        .byte 0
${ABI_UNDERSCORE}knMicInteger_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/mic/knMicInteger.h"
        .byte 0
${ABI_UNDERSCORE}knMicTernary_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/mic/knMicTernary.h"
        .byte 0
${ABI_UNDERSCORE}knMicGather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/mic/knMicGather.h"
        .byte 0
${ABI_UNDERSCORE}knMicScatter_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/mic/knMicScatter.h"
        .byte 0
${ABI_UNDERSCORE}knMicOStream_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/dump/mic/knMicOStream.h"
        .byte 0

