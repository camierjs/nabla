###############
# Now testing #
###############
enable_testing()


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
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/nabla.unf ${CMAKE_CURRENT_BINARY_DIR}/nabla.unf)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/sod_triangles.unf ${CMAKE_CURRENT_BINARY_DIR}/sod_triangles.unf)


#######################################
# NABLA TEST MACRO FOR ARCANE BACKEND #
#######################################
function(get_additional_nabla_files_and_set_Makefile name)
  #info("${VT100_FG_GREEN}WORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}/${name}${VT100_RESET}")

  execute_process(COMMAND find . -name *.n -printf "%f "
    COMMAND sed -e "s/\\(${name}.n\\)//g"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${name}
    OUTPUT_VARIABLE ADDITIONAL_NABLA_FILES
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  #info("${VT100_FG_GREEN}get_additional_nabla_files: ${ADDITIONAL_NABLA_FILES}${VT100_RESET}")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla ${CMAKE_CURRENT_BINARY_DIR}/${name}/Makefile.${backend})
  separate_arguments(ADDITIONAL_NABLA_FILES)
  foreach(file ${name}.n ${ADDITIONAL_NABLA_FILES})
    #info("${VT100_FG_GREEN}file=${file}${VT100_RESET}") 
    execute_process(COMMAND /bin/ln -fs ${CMAKE_CURRENT_SOURCE_DIR}/${name}/${file} . 
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endforeach()
endfunction()

function(nabla_add_arcane_test_with_additional_defines name nb_proc cmd additional_defines)
  info("Adding ctest ${VT100_FG_WHITE}nabla_${name}_${VT100_FG_BLUE}arcane${VT100_FG_WHITE}_${nb_proc}proc_${cmd}${VT100_RESET}")
  set(backend arcane)
  get_additional_nabla_files_and_set_Makefile(${name})
  add_test(NAME nabla_arcane_${name}_${cmd}_${nb_proc} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}
	 COMMAND /bin/sh -c "export ADDITIONAL_DEFINES=-D${additional_defines} && ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/Makefile.arcane ${cmd}${nb_proc}")
endfunction(nabla_add_arcane_test_with_additional_defines name nb_proc cmd additional_defines)

function(NABLA_ADD_ARCANE_TEST name nb_proc cmd)
   nabla_add_arcane_test_with_additional_defines(${name} ${nb_proc} ${cmd} NO_ADDITIONAL_DEFINES)
endfunction(NABLA_ADD_ARCANE_TEST)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.arcane ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.arcane)


######################################
# NABLA TEST MACRO FOR OKINA BACKEND #
######################################
function(NABLA_ADD_OKINA_TEST name nb_proc cmd simd parallel)
  info("Adding ctest ${VT100_FG_WHITE}nabla_${name}_${VT100_FG_YELLOW}okina${VT100_FG_WHITE}_${nb_proc}proc_${cmd}_${simd}_${parallel}${VT100_RESET}")
  set(backend okina)
  get_additional_nabla_files_and_set_Makefile(${name})
  add_test(NAME nabla_okina_${name}_${cmd}_${nb_proc}_${simd}_${parallel} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}
	 COMMAND /bin/sh -c "PARALLEL=${parallel} SIMD=${simd} ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/Makefile.okina ${cmd}${nb_proc}")
endfunction(NABLA_ADD_OKINA_TEST)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.okina ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.okina)
 

#####################################
# NABLA TEST MACRO FOR CUDA BACKEND #
#####################################
function(NABLA_ADD_CUDA_TEST name nb_proc cmd)
  info("Adding ctest ${VT100_FG_WHITE}nabla_${name}_${VT100_FG_GREEN}cuda${VT100_FG_WHITE}${VT100_RESET}")
  set(backend cuda)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla ${CMAKE_CURRENT_BINARY_DIR}/${name}/Makefile.${backend})
  add_test(NAME nabla_cuda_${name}_${cmd}_${nb_proc} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}
	 COMMAND ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/Makefile.cuda ${cmd}${nb_proc})
endfunction(NABLA_ADD_CUDA_TEST)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.cuda ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.cuda)



########################################
# Now scan tests directory to add some # 
########################################
execute_process(COMMAND /bin/ls -1XF 
                COMMAND sed -e "/.*\\//!d;s/\\///g"
                COMMAND tr \\n " "
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE RESULT_NABLA_TESTS
                OUTPUT_VARIABLE NABLA_TESTS
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if (${RESULT_NABLA_TESTS})
  message(SEND_ERROR "Error: Could not load tests directories!")
endif (${RESULT_NABLA_TESTS})
separate_arguments(NABLA_TESTS)
foreach(tst IN ITEMS ${NABLA_TESTS})
  string(REGEX MATCH ".*_aleph_index$" ALEPH_INDEX ${tst})
  if (ALEPH_INDEX)
    nabla_add_arcane_test_with_additional_defines(${tst} 1 gen ALEPH_INDEX)
  else (ALEPH_INDEX)
    nabla_add_arcane_test(${tst} 1 gen)
  endif (ALEPH_INDEX)
endforeach(tst)


##########################
# NABLA TESTS FOR ARCANE #
##########################
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



foreach(simd std sse avx avx2 mic)
  foreach(parallel omp cilk)
    #info("Adding ctest ${VT100_FG_WHITE}nabla_${simd}_${parallel}${VT100_RESET}")
    nabla_add_okina_test(lulesh_mic 1 gen ${simd} ${parallel})
  endforeach(parallel)
endforeach(simd)


#nabla_add_okina_test(lulesh_mic 1 gen std omp)
#nabla_add_okina_test(lulesh_mic 1 gen sse omp)
#nabla_add_okina_test(lulesh_mic 1 gen avx omp)
#nabla_add_okina_test(lulesh_mic 1 gen avx2 omp)
#nabla_add_okina_test(lulesh_mic 1 gen std cilk)
#nabla_add_okina_test(lulesh_mic 1 gen sse cilk)
#nabla_add_okina_test(lulesh_mic 1 gen avx cilk)
#nabla_add_okina_test(lulesh_mic 1 gen avx2 cilk)

#nabla_add_okina_test(lulesh_mic 1 run std omp)
#nabla_add_okina_test(lulesh_mic 1 run sse omp)
#nabla_add_okina_test(lulesh_mic 1 run avx omp)
##nabla_add_okina_test(lulesh_mic 1 run avx2 omp)
##nabla_add_okina_test(lulesh_mic 1 run sse cilk)
##nabla_add_okina_test(lulesh_mic 1 run avx cilk)


nabla_add_cuda_test(mhydro 1 gen)

if(CUDA_INCLUDE_DIRS)
  nabla_add_arcane_test(mhydro 1 hyoda)
  nabla_add_arcane_test(mhydro 4 hyoda)
  nabla_add_cuda_test(mhydro 1 run)
endif(CUDA_INCLUDE_DIRS)
