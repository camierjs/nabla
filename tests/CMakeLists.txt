###############
# Now testing #
###############
enable_testing()
set(ARCANE_TESTS true)
set(OKINA_TESTS true)


#########################
# Third Party Libraries #
#########################
set(MPI_PATH /usr/local/openmpi-1.5.3/gnu)
set(ARCANE_ROOT /usr/local/arcane/gcc47-mpi/1.22.4)
set(ARCANE_CXX /usr/local/opendev1/gcc/gcc/4.8.1/bin/g++)

set(SLOOP_INC /usr/local/nec/share/include/sloop-5.5.4)
set(HYPRE_INC /usr/local/opendev1/gcc/hypre/2.9.0b-ompi15)
set(SUPERLU_DIR /usr/local/opendev1/icc/superlu/superlu_DIST_3.2-ompi15)
set(SUPERLU_LIB "-L${SUPERLU_DIR}/lib -lsuperlu_dist_3.2 -Wl,-rpath,${SUPERLU_DIR}/lib")
set(GMP_DIR /usr/local/opendev1/gcc/gmp/5.0.5)
set(GMP_LIB "-L${GMP_DIR}/lib -lgmp -Wl,-rpath,${GMP_DIR}/lib")
set(CEA_LIB /usr/local/sr/lib/libcea_user.a)
#set(ICET_DIR /usr/local/opendev1/gcc/iceT/2.1.1-ompi15)
#set(OSMESA_DIR /usr/local/opendev1/gcc/mesa/7.10.3)
#set(TBB_DIR /usr/local/opendev1/gcc/tbb/4.0.1)


##########
# Meshes #
##########
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/unf/nabla.unf ${CMAKE_CURRENT_BINARY_DIR}/nabla.unf)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/unf/sod_triangles.unf ${CMAKE_CURRENT_BINARY_DIR}/sod_triangles.unf)


#######################################
# NABLA TEST MACRO FOR ARCANE BACKEND #
#######################################
function(nabla_add_arcane_test_with_additional_defines name nb_proc cmd additional_defines)
	info("Adding ctest ${VT100_FG_WHITE}nabla_${name}_${VT100_FG_YELLOW}arcane${VT100_FG_WHITE}_${nb_proc}proc_${cmd}${VT100_RESET}")
   set(backend arcane)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla ${CMAKE_CURRENT_BINARY_DIR}/${name}/Makefile.${backend})
	add_test(NAME nabla_${name}_arcane_${nb_proc}proc_${cmd} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}
	  COMMAND /bin/sh -c "export TGT=${name} && export ADDITIONAL_DEFINES=-D${additional_defines} && ${CMAKE_MAKE_PROGRAM} -f ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.arcane ${cmd}${nb_proc}")
 endfunction(nabla_add_arcane_test_with_additional_defines name nb_proc cmd additional_defines)

function(NABLA_ADD_ARCANE_TEST name nb_proc cmd)
   nabla_add_arcane_test_with_additional_defines(${name} ${nb_proc} ${cmd} NO_ADDITIONAL_DEFINES)
endfunction(NABLA_ADD_ARCANE_TEST)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.arcane ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.arcane)


######################################
# NABLA TEST MACRO FOR OKINA BACKEND #
######################################
function(NABLA_ADD_OKINA_TEST name nb_proc cmd)
	info("Adding ctest ${VT100_FG_WHITE}nabla_${name}_${VT100_FG_YELLOW}okina${VT100_FG_WHITE}_${nb_proc}proc_${cmd}${VT100_RESET}")
   set(backend okina)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla ${CMAKE_CURRENT_BINARY_DIR}/${name}/Makefile.${backend})
	add_test(NAME nabla_${name}_okina_${nb_proc}proc_${cmd} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}
	  COMMAND /bin/sh -c "${CMAKE_MAKE_PROGRAM} -f ${CMAKE_CURRENT_BINARY_DIR}/${name}/Makefile.okina ${cmd}${nb_proc}")
endfunction(NABLA_ADD_OKINA_TEST)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.okina ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.okina)


#####################################
# NABLA TEST MACRO FOR CUDA BACKEND #
#####################################
if(CUDA_INCLUDE_DIRS)
function(NABLA_ADD_CUDA_TEST name nb_proc)
	info("Adding ctest ${VT100_FG_CYAN}nabla_${name}_${VT100_FG_GREEN}cuda${VT100_FG_CYAN}_${nb_proc}proc${VT100_RESET}")
	add_test(NAME nabla_${name}_cuda_${nb_proc}proc
	   		WORKING_DIRECTORY ${ARCANE_NABLA_TEST_PATH}
				COMMAND ${CMAKE_MAKE_PROGRAM} -B -f Makefile.nabla.cuda g${nb_proc} ${name})
endfunction(NABLA_ADD_CUDA_TEST)
configure_file(${ARCANE_CEA_SOURCE_PATH}/src/arcane/nabla/tests/Makefile.nabla.cuda ${ARCANE_NABLA_TEST_PATH}/Makefile.nabla.cuda)
endif(CUDA_INCLUDE_DIRS)


##########################
# NABLA TESTS FOR ARCANE #
##########################
if (ARCANE_TESTS)
  nabla_add_arcane_test(cartesian 1 run)
  nabla_add_arcane_test(materials 1 run)
  nabla_add_arcane_test(ddfv 1 run)
  nabla_add_arcane_test(ddfv 4 run)
  nabla_add_arcane_test_with_additional_defines(poisson_diff 1 run ALEPH_INDEX)
  nabla_add_arcane_test_with_additional_defines(poisson_diff 4 run ALEPH_INDEX)
  nabla_add_arcane_test_with_additional_defines(poisson_ddfv 1 run ALEPH_INDEX)
  nabla_add_arcane_test_with_additional_defines(poisson_ddfv 4 run ALEPH_INDEX)
  nabla_add_arcane_test_with_additional_defines(schrodinger 1 run ALEPH_INDEX)
  nabla_add_arcane_test_with_additional_defines(schrodinger 4 run ALEPH_INDEX)
  nabla_add_arcane_test(mhydro 1 run)
  nabla_add_arcane_test(mhydro 4 run)
  nabla_add_arcane_test(shydro 1 run)
  nabla_add_arcane_test(shydro 4 run)
  nabla_add_arcane_test(lulesh 1 run)
  nabla_add_arcane_test(lulesh 4 run)
  nabla_add_arcane_test(comd 1 run)
  nabla_add_arcane_test(mc1d 1 run)
  nabla_add_arcane_test(mctb 1 run)
  nabla_add_arcane_test(sethi 1 run)
  nabla_add_arcane_test(sethi 4 run)
  nabla_add_arcane_test(glace 1 run)
  nabla_add_arcane_test(glace 4 run)
  nabla_add_arcane_test(fermat 1 run)
  nabla_add_arcane_test(fermat 1 runc)
  nabla_add_arcane_test(fermat 4 run)
  nabla_add_arcane_test(gad 1 run)
endif(ARCANE_TESTS)


if(OKINA_TESTS)
  nabla_add_okina_test(lulesh_mic 1 run)
endif(OKINA_TESTS)


if(CUDA_INCLUDE_DIRS)
  nabla_add_arcane_test(mhydro 1 hyoda)
  nabla_add_arcane_test(mhydro 4 hyoda)
  nabla_add_cuda_test(mhydro 1 run)
endif(CUDA_INCLUDE_DIRS)
