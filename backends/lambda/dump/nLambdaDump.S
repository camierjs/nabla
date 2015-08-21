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
        .section ".rodata"
        .global lambdaDbg_h
        .global lambdaMsh1D_c
        .global lambdaMsh3D_c
        .global lambdaMth_h
        
        .global lambdaStdReal_h
        .global lambdaStdReal3_h
        .global lambdaStdInteger_h
        .global lambdaStdTernary_h
        .global lambdaStdGather_h
        .global lambdaStdScatter_h
        .global lambdaStdOStream_h
        
lambdaDbg_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaDbg.h"
        .byte 0
lambdaMsh1D_c: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaMsh1D.c"
        .byte 0
lambdaMsh3D_c: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaMsh3D.c"
        .byte 0
lambdaMth_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaMth.h"
        .byte 0

 
        
lambdaStdReal_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaStdReal.h"
        .byte 0
lambdaStdReal3_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaStdReal3.h"
        .byte 0
lambdaStdInteger_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaStdInteger.h"
        .byte 0
lambdaStdTernary_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaStdTernary.h"
        .byte 0
lambdaStdGather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaStdGather.h"
        .byte 0
lambdaStdScatter_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaStdScatter.h"
        .byte 0
lambdaStdOStream_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/nLambdaStdOStream.h"
        .byte 0

