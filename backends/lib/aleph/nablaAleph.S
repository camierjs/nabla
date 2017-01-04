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
   .globl ${ABI_UNDERSCORE}nablaAleph_h
   
   .globl ${ABI_UNDERSCORE}AlephStd_h
   .globl ${ABI_UNDERSCORE}AlephStd_c

   .globl ${ABI_UNDERSCORE}Aleph_h
   .globl ${ABI_UNDERSCORE}AlephTypesSolver_h
   .globl ${ABI_UNDERSCORE}AlephParams_h
   .globl ${ABI_UNDERSCORE}AlephVector_h
   .globl ${ABI_UNDERSCORE}AlephMatrix_h
   .globl ${ABI_UNDERSCORE}AlephKernel_h
   .globl ${ABI_UNDERSCORE}AlephOrdering_h
   .globl ${ABI_UNDERSCORE}AlephIndexing_h
   .globl ${ABI_UNDERSCORE}AlephTopology_h
	.globl ${ABI_UNDERSCORE}AlephInterface_h
   
   .globl ${ABI_UNDERSCORE}IAlephFactory_h
   
${ABI_UNDERSCORE}nablaAleph_h: 
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/nablaAleph.h"
   .byte 0

   
${ABI_UNDERSCORE}AlephStd_c: 
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephStd.cc"
   .byte 0
${ABI_UNDERSCORE}AlephStd_h: 
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephStd.h"
   .byte 0
   
${ABI_UNDERSCORE}Aleph_h: 
	.incbin "${NABLA_SOURCE_DIR}/aleph/Aleph.h"
   .byte 0
${ABI_UNDERSCORE}AlephTypesSolver_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephTypesSolver.h"
   .byte 0

${ABI_UNDERSCORE}AlephParams_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephParams.h"
   .byte 0
${ABI_UNDERSCORE}AlephVector_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephVector.h"
   .byte 0
${ABI_UNDERSCORE}AlephMatrix_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephMatrix.h"
   .byte 0
${ABI_UNDERSCORE}AlephKernel_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephKernel.h"
   .byte 0
${ABI_UNDERSCORE}AlephOrdering_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephOrdering.h"
   .byte 0
${ABI_UNDERSCORE}AlephIndexing_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephIndexing.h"
   .byte 0
${ABI_UNDERSCORE}AlephTopology_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephTopology.h"
   .byte 0
${ABI_UNDERSCORE}AlephInterface_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/AlephInterface.h"
   .byte 0
   
${ABI_UNDERSCORE}IAlephFactory_h:
	.incbin "${NABLA_SOURCE_DIR}/aleph/IAlephFactory.h"
   .byte 0
