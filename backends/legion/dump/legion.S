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
   .globl ${ABI_UNDERSCORE}legion_root_path

   .globl ${ABI_UNDERSCORE}legion_args_rg
   .globl ${ABI_UNDERSCORE}legion_common_rg
   .globl ${ABI_UNDERSCORE}legion_compile_rg
  	.globl ${ABI_UNDERSCORE}legion_config_rg
  	.globl ${ABI_UNDERSCORE}legion_data_rg
   .globl ${ABI_UNDERSCORE}legion_distri_rg
   .globl ${ABI_UNDERSCORE}legion_geom_rg
   .globl ${ABI_UNDERSCORE}legion_hacks_rg
   .globl ${ABI_UNDERSCORE}legion_init_rg
   .globl ${ABI_UNDERSCORE}legion_input_rg
   .globl ${ABI_UNDERSCORE}legion_math_rg
   .globl ${ABI_UNDERSCORE}legion_mesh_rg
   .globl ${ABI_UNDERSCORE}legion_rg
   .globl ${ABI_UNDERSCORE}legion_tools_rg

   .globl ${ABI_UNDERSCORE}legion_cc
   .globl ${ABI_UNDERSCORE}legion_h

 	.globl ${ABI_UNDERSCORE}makefile

   .globl ${ABI_UNDERSCORE}pennant_data_rg
   .globl ${ABI_UNDERSCORE}pennant_kernels_init_rg
   .globl ${ABI_UNDERSCORE}pennant_kernels_loop_rg
   .globl ${ABI_UNDERSCORE}pennant_main_init_rg
   .globl ${ABI_UNDERSCORE}pennant_main_loop_rg
   .globl ${ABI_UNDERSCORE}pennant_valid_rg
   
   .globl ${ABI_UNDERSCORE}sedovsmall_pnt
   .globl ${ABI_UNDERSCORE}sedovsmall_xy_std


${ABI_UNDERSCORE}legion_root_path:
	.string "${LEGION_ROOT_PATH}"
   .byte 0
   
${ABI_UNDERSCORE}legion_args_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_args.rg"
   .byte 0
${ABI_UNDERSCORE}legion_common_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_common.rg"
   .byte 0
${ABI_UNDERSCORE}legion_compile_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_compile.rg"
   .byte 0
${ABI_UNDERSCORE}legion_config_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_config.rg"
   .byte 0
${ABI_UNDERSCORE}legion_data_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_data.rg"
   .byte 0
${ABI_UNDERSCORE}legion_distri_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_distri.rg"
   .byte 0
${ABI_UNDERSCORE}legion_geom_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_geom.rg"
   .byte 0
${ABI_UNDERSCORE}legion_hacks_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_hacks.rg"
   .byte 0
${ABI_UNDERSCORE}legion_init_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_init.rg"
   .byte 0
${ABI_UNDERSCORE}legion_input_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_input.rg"
   .byte 0
${ABI_UNDERSCORE}legion_math_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_math.rg"
   .byte 0
${ABI_UNDERSCORE}legion_mesh_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_mesh.rg"
   .byte 0
${ABI_UNDERSCORE}legion_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion.rg"
   .byte 0
${ABI_UNDERSCORE}legion_tools_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion_tools.rg"
   .byte 0

${ABI_UNDERSCORE}legion_cc:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion.cc"
   .byte 0
${ABI_UNDERSCORE}legion_h:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/legion.h"
   .byte 0

${ABI_UNDERSCORE}makefile:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/legion/makefile"
   .byte 0

${ABI_UNDERSCORE}pennant_data_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/pennant/pennant_data.rg"
   .byte 0
${ABI_UNDERSCORE}pennant_kernels_init_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/pennant/pennant_kernels_init.rg"
   .byte 0
${ABI_UNDERSCORE}pennant_kernels_loop_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/pennant/pennant_kernels_loop.rg"
   .byte 0
${ABI_UNDERSCORE}pennant_main_init_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/pennant/pennant_main_init.rg"
   .byte 0
${ABI_UNDERSCORE}pennant_main_loop_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/pennant/pennant_main_loop.rg"
   .byte 0
${ABI_UNDERSCORE}pennant_valid_rg:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/pennant/pennant_valid.rg"
   .byte 0
   
${ABI_UNDERSCORE}sedovsmall_pnt:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/pennant/sedovsmall.pnt"
   .byte 0
${ABI_UNDERSCORE}sedovsmall_xy_std:
	.incbin "${CMAKE_CURRENT_SOURCE_DIR}/pennant/sedovsmall.xy.std"
   .byte 0
