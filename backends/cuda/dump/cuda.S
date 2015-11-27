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
   //.section ".rodata"
   .globl ${ABI_UNDERSCORE}cuTypes_h
   .globl ${ABI_UNDERSCORE}cuDebug_h
   .globl ${ABI_UNDERSCORE}cuError_h
   .globl ${ABI_UNDERSCORE}cuExtra_h
   .globl ${ABI_UNDERSCORE}cuMeshs_h
   .globl ${ABI_UNDERSCORE}cuItems_h

${ABI_UNDERSCORE}cuTypes_h:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/cuTypes.h"
   .byte 0
${ABI_UNDERSCORE}cuDebug_h:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/cuDebug.h"
   .byte 0
${ABI_UNDERSCORE}cuError_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/cuError.h"
   .byte 0
${ABI_UNDERSCORE}cuExtra_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/cuExtra.h"
   .byte 0   
${ABI_UNDERSCORE}cuMeshs_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/cuMeshs.h"
   .byte 0   
${ABI_UNDERSCORE}cuItems_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/cuItems.h"
   .byte 0   
         
