###############
# Now testing #
###############
enable_testing()
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}TESTS${VT100_RESET}")
include(${NABLA_SOURCE_DIR}/CMake/CMakeTPL.txt)

##########
# Meshes #
##########
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different 
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/nabla.unf ${CMAKE_CURRENT_BINARY_DIR}/mesh/nabla.unf)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/thex_mesh.unf ${CMAKE_CURRENT_BINARY_DIR}/mesh/thex_mesh.unf)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different 
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/sod_triangles.unf ${CMAKE_CURRENT_BINARY_DIR}/mesh/sod_triangles.unf)


#######################################
# NABLA TEST MACRO FOR ARCANE BACKEND #
#######################################

###############################################
# get_additional_nabla_files_and_set_Makefile #
function(get_additional_nabla_files_and_set_Makefile name backend path)
   #info("${VT100_FG_GREEN}WORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}/${name}${VT100_RESET}")
   execute_process(COMMAND find . -name *.n
		             COMMAND sed -e "s/\\.\\///g"
		             COMMAND tr "\\n" " "
                   COMMAND sed -e "s/\\(${name}\\.n\\)//g"
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${name}
                   OUTPUT_VARIABLE ADDITIONAL_NABLA_FILES
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
  #info("${VT100_FG_GREEN}get_additional_nabla_files: ${ADDITIONAL_NABLA_FILES}${VT100_RESET}")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${path}/Makefile)
  separate_arguments(ADDITIONAL_NABLA_FILES)
  foreach(file ${name}.n ${ADDITIONAL_NABLA_FILES})
    #info("${VT100_FG_GREEN}file=${file}${VT100_RESET}") 
    execute_process(COMMAND /bin/ln -fs ${CMAKE_CURRENT_SOURCE_DIR}/${name}/${file} . 
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${path}
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endforeach()
endfunction()

#################################################
# nabla_add_arcane_test_with_additional_defines #
function(nabla_add_arcane_test_with_additional_defines name nb_proc cmd additional_defines)
  info("Adding ctest ${VT100_FG_WHITE}nabla_${name}_${VT100_FG_CYAN}arcane${VT100_FG_WHITE}_${nb_proc}proc_${cmd}${VT100_RESET}")
  set(backend arcane)
  set(subpath ${nb_proc})
  get_additional_nabla_files_and_set_Makefile(${name} ${backend} ${subpath})
  add_test(NAME nabla_arcane_${name}_${cmd}_${nb_proc} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}
	 COMMAND /bin/sh -c "export ADDITIONAL_DEFINES=-D${additional_defines} && ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}/Makefile ${cmd}${nb_proc}")
endfunction()

#########################
# nabla_add_arcane_test #
function(nabla_add_arcane_test name nb_proc cmd)
   nabla_add_arcane_test_with_additional_defines(${name} ${nb_proc} ${cmd} NO_ADDITIONAL_DEFINES)
endfunction()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.arcane ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.arcane)


######################################
# NABLA TEST MACRO FOR OKINA BACKEND #
######################################
function(NABLA_ADD_OKINA_TEST name nb_proc cmd simd parallel)
  info("Adding ctest ${VT100_FG_WHITE}nabla_${name}_${VT100_FG_YELLOW}okina${VT100_FG_WHITE}_${nb_proc}proc_${VT100_FG_YELLOW}${cmd}${VT100_FG_WHITE}_${simd}_${parallel}${VT100_RESET}")
  set(backend okina)
  set(subpath ${nb_proc}/${cmd}/${simd}/${parallel})
  get_additional_nabla_files_and_set_Makefile(${name} ${backend} ${subpath})
  add_test(NAME nabla_okina_${name}_${cmd}_${nb_proc}_${simd}_${parallel} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}
	 COMMAND /bin/sh -c "PARALLEL=${parallel} SIMD=${simd} ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}/Makefile ${cmd}${nb_proc}")
endfunction(NABLA_ADD_OKINA_TEST)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.okina ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.okina)


#######################################
# NABLA TEST MACRO FOR LAMBDA BACKEND #
#######################################
function(NABLA_ADD_LAMBDA_TEST name nb_proc cmd parallel)
  info("Adding ctest ${VT100_FG_WHITE}nabla_${name}_${VT100_FG_MAGENTA}lambda${VT100_FG_WHITE}_${nb_proc}proc_${VT100_FG_YELLOW}${cmd}${VT100_FG_WHITE}_${parallel}${VT100_RESET}")
  set(backend lambda)
  set(subpath ${nb_proc}/${cmd}/${parallel})
  get_additional_nabla_files_and_set_Makefile(${name} ${backend} ${subpath})
  add_test(NAME nabla_lambda_${name}_${cmd}_${nb_proc}_${parallel} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}
	 COMMAND /bin/sh -c "PARALLEL=${parallel} ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}/Makefile ${cmd}${nb_proc}")
endfunction(NABLA_ADD_LAMBDA_TEST)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.lambda ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.lambda)
 

#####################################
# NABLA TEST MACRO FOR CUDA BACKEND #
#####################################
function(NABLA_ADD_CUDA_TEST name nb_proc cmd)
  info("Adding ctest ${VT100_FG_WHITE}nabla_${name}_${VT100_FG_GREEN}cuda${VT100_FG_WHITE}_${VT100_FG_YELLOW}${cmd}${VT100_RESET}")
  set(backend cuda)
  get_additional_nabla_files_and_set_Makefile(${name} ${backend} .)
  add_test(NAME nabla_cuda_${name}_${cmd}_${nb_proc} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}
	 COMMAND ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/Makefile ${cmd}${nb_proc})
endfunction(NABLA_ADD_CUDA_TEST)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.cuda ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.cuda)



########################################
# Now scan tests directory to add them #
# Avoid '.' and 'mesh'                 #
########################################
execute_process(COMMAND find . -maxdepth 1 -type d -name [^\\.]*[^\(mesh\)]*
		          COMMAND sed -e "s/\\.\\// /g"
                COMMAND tr "\\n" " "
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE RESULT_NABLA_TESTS
                OUTPUT_VARIABLE NABLA_TESTS
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if (${RESULT_NABLA_TESTS})
  message(SEND_ERROR "Error: Could not load tests directories!")
endif (${RESULT_NABLA_TESTS})
separate_arguments(NABLA_TESTS)
#info("Loading NABLA_TESTS: ${NABLA_TESTS}${VT100_RESET}")

foreach(tst ${NABLA_TESTS})
  #info("\tFocusing on test ${tst}")
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
if (ARCANE_FOUND)
  # No ICartesianMesh out of CEA # nabla_add_arcane_test(gad 1 run)
  nabla_add_arcane_test(comd 1 run)
  nabla_add_arcane_test(mhydro 1 run)
  # OK, mais trop de temps: nabla_add_arcane_test(mhydro 3 run)
  # OK, mais trop de temps: nabla_add_arcane_test(mhydro 4 run)
  nabla_add_arcane_test(glace 1 run)
  # No ICartesianMesh out of CEA # nabla_add_arcane_test(lulesh 1 run)
  #nabla_add_arcane_test(lulesh 4 run)
  #nabla_add_arcane_test(lulesh 8 run)
if (HYPRE_FOUND)
  nabla_add_arcane_test_with_additional_defines(schrodinger 1 run ALEPH_INDEX)
  # OK, mais trop de temps: nabla_add_arcane_test_with_additional_defines(schrodinger 4 run ALEPH_INDEX)
  nabla_add_arcane_test_with_additional_defines(pDDFV 1 run ALEPH_INDEX)
  # OK, mais trop de temps:nabla_add_arcane_test_with_additional_defines(pDDFV 4 run ALEPH_INDEX)
endif (HYPRE_FOUND)
  # No ICartesianMesh out of CEA # nabla_add_arcane_test(sethi 1 run)
  # No ICartesianMesh out of CEA # nabla_add_arcane_test(sethi 4 run)
  nabla_add_arcane_test(shydro 1 run)
  # OK, mais trop de temps:nabla_add_arcane_test(shydro 3 run)
  # OK, mais trop de temps:nabla_add_arcane_test(shydro 4 run)
endif(ARCANE_FOUND)


#########################
# NABLA TESTS FOR OKINA #
#########################
set(SIMD_MODES "std sse avx avx2") #mic
set(PARALLEL_MODES "seq omp") #cilk
if(APPLE)
  set(SIMD_MODES "std sse")
  set(PARALLEL_MODES "seq")
  endif(APPLE)
  
separate_arguments(SIMD_MODES)
separate_arguments(PARALLEL_MODES)

foreach(parallel seq omp cilk)
  foreach(simd std sse avx avx2)
    nabla_add_okina_test(lulesh 1 gen ${simd} ${parallel})
  endforeach()
endforeach()

foreach(parallel ${PARALLEL_MODES})
  foreach(simd ${SIMD_MODES})
    nabla_add_okina_test(lulesh 1 run ${simd} ${parallel})
    nabla_add_okina_test(lulesh 2 run ${simd} ${parallel})
  endforeach()
endforeach()

nabla_add_okina_test(upwind 1 gen std seq)
nabla_add_okina_test(upwind 1 run std seq)

nabla_add_okina_test(upwindAP 1 gen std seq)
nabla_add_okina_test(upwindAP 1 run std seq)


##########################
# NABLA TESTS FOR LAMBDA #
##########################
nabla_add_lambda_test(lulesh 1 gen seq)
nabla_add_lambda_test(lulesh 1 run seq)
nabla_add_lambda_test(lulesh 1 run omp)

nabla_add_lambda_test(upwindAP 1 gen seq)
nabla_add_lambda_test(upwindAP 1 run seq)


########################
# NABLA TESTS FOR CUDA #
########################
nabla_add_cuda_test(gram 1 gen)
nabla_add_cuda_test(lulesh 1 gen)
nabla_add_cuda_test(mhydro 1 gen)
if(CUDA_FOUND)
  #nabla_add_cuda_test(mhydro 1 run)
  nabla_add_cuda_test(lulesh 1 run)
endif()
