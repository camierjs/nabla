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
  info("Adding ctest nabla_${VT100_FG_RED}${cmd}${VT100_RESET}_${VT100_FG_CYAN}arcane${VT100_RESET}_${VT100_FG_WHITE}${name}${VT100_RESET}_${nb_proc}_std_mpi")
  set(backend arcane)
  set(subpath ${nb_proc}/${cmd})
  get_additional_nabla_files_and_set_Makefile(${name} ${backend} ${subpath})
  add_test(NAME nabla_${cmd}_arcane_${name}_${nb_proc}_std_mpi
				WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}
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
  info("Adding ctest nabla_${VT100_FG_RED}${cmd}${VT100_RESET}_${VT100_FG_YELLOW}okina${VT100_RESET}_${VT100_FG_WHITE}${name}${VT100_RESET}_${nb_proc}_${simd}_${parallel}")
  set(backend okina)
  set(subpath ${nb_proc}/${cmd}/${simd}/${parallel})
  get_additional_nabla_files_and_set_Makefile(${name} ${backend} ${subpath})
  add_test(NAME nabla_${cmd}_okina_${name}_${nb_proc}_${simd}_${parallel}
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}
		COMMAND /bin/sh -c "PARALLEL=${parallel} SIMD=${simd} ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}/Makefile ${cmd}${nb_proc}")
endfunction(NABLA_ADD_OKINA_TEST)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.okina ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.okina)


#######################################
# NABLA TEST MACRO FOR LAMBDA BACKEND #
#######################################
function(nabla_add_lambda_test_with_defines name nb_proc cmd parallel defines nb_particles)
info("Adding ctest nabla_${VT100_FG_RED}${cmd}${VT100_RESET}_${VT100_FG_MAGENTA}lambda${VT100_RESET}_${VT100_FG_WHITE}${name}${VT100_RESET}_${nb_proc}_std_${parallel}")
  set(backend lambda)
  set(subpath ${nb_proc}/${cmd}/${parallel})
  get_additional_nabla_files_and_set_Makefile(${name} ${backend} ${subpath})
  add_test(NAME nabla_${cmd}_lambda_${name}_${nb_proc}_std_${parallel}
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}
		COMMAND /bin/sh -c "ADDITIONAL_DEFINES='${defines} -DNB_PARTICLES=${nb_particles}' PARALLEL=${parallel} ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}/Makefile ${cmd}${nb_proc}")
endfunction(nabla_add_lambda_test_with_defines)

function(nabla_add_lambda_test name nb_proc cmd parallel)
   nabla_add_lambda_test_with_defines(${name} ${nb_proc} ${cmd} ${parallel} -DALEPH_INDEX 0)
endfunction(nabla_add_lambda_test)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.lambda ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.lambda)


########################################
# NABLA TEST MACRO FOR GENERIC BACKEND #
########################################
function(nabla_add_test_with_defines backend name nb_proc cmd parallel defines nb_particles)
info("Adding ctest nabla_${VT100_FG_RED}${cmd}${VT100_RESET}_${VT100_FG_MAGENTA}${backend}${VT100_RESET}_${VT100_FG_WHITE}${name}${VT100_RESET}_${nb_proc}_std_${parallel}")
  set(backend ${backend})
  set(subpath ${nb_proc}/${cmd}/${parallel})
  get_additional_nabla_files_and_set_Makefile(${name} ${backend} ${subpath})
  add_test(NAME nabla_${cmd}_${backend}_${name}_${nb_proc}_std_${parallel}
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}
		COMMAND /bin/sh -c "ADDITIONAL_DEFINES='${defines} -DNB_PARTICLES=${nb_particles}' PARALLEL=${parallel} ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}/Makefile ${cmd}${nb_proc}")
endfunction(nabla_add_test_with_defines)

function(nabla_add_test backend name nb_proc cmd parallel)
   nabla_add_test_with_defines(${backend} ${name} ${nb_proc} ${cmd} ${parallel} -DALEPH_INDEX 0)
endfunction(nabla_add_test)


##################################
# NABLA TESTS FOR KOKKOS BACKEND #
##################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.kokkos ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.kokkos)


#####################################
# NABLA TEST MACRO FOR CUDA BACKEND #
#####################################
function(NABLA_ADD_CUDA_TEST name nb_proc cmd)
info("Adding ctest  nabla_${VT100_FG_RED}${cmd}${VT100_RESET}_${VT100_FG_GREEN}cuda${VT100_RESET}_${VT100_FG_WHITE}${name}${VT100_RESET}_${nb_proc}_warp_smp")
  set(backend cuda)
  set(subpath ${nb_proc}/${cmd})
  get_additional_nabla_files_and_set_Makefile(${name} ${backend} ${subpath})
  add_test(NAME nabla_${cmd}_cuda_${name}_${nb_proc}_std_smp
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}
		COMMAND ${CMAKE_MAKE_PROGRAM} -kBf ${CMAKE_CURRENT_BINARY_DIR}/${name}/${backend}/${subpath}/Makefile ${cmd}${nb_proc})
endfunction(NABLA_ADD_CUDA_TEST)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.nabla.cuda ${CMAKE_CURRENT_BINARY_DIR}/Makefile.nabla.cuda)



########################################
# Now scan tests directory to add them #
# Avoid '.', 'mesh' and ns4n           #
########################################
execute_process(COMMAND find . -maxdepth 1 -type d -name [^\\.]*[^\(mesh\)]*[^\(ns4n\)]*
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
  nabla_add_arcane_test(hlt 1 run)
  # No ICartesianMesh out of CEA # nabla_add_arcane_test(gad 1 run)
  nabla_add_arcane_test(anyItem 1 run)
  nabla_add_arcane_test(comd 1 run)
  nabla_add_arcane_test(mhydro 1 run)
  # OK, mais trop de temps: nabla_add_arcane_test(mhydro 3 run)
  # OK, mais trop de temps: nabla_add_arcane_test(mhydro 4 run)
  nabla_add_arcane_test(glace 1 run)
  nabla_add_arcane_test(glace2D 1 run)
  nabla_add_arcane_test(glc 1 run)
  # No ICartesianMesh out of CEA # nabla_add_arcane_test(lulesh 1 run)
  #nabla_add_arcane_test(lulesh 1 run)
  #nabla_add_arcane_test(lulesh 8 run)
if (HYPRE_FOUND)
  nabla_add_arcane_test_with_additional_defines(darcy 1 run ALEPH_INDEX)
  nabla_add_arcane_test_with_additional_defines(darcy 4 run ALEPH_INDEX)
  nabla_add_arcane_test_with_additional_defines(heat 1 run ALEPH_INDEX)
  #nabla_add_arcane_test_with_additional_defines(heat 4 run ALEPH_INDEX)
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

nabla_add_okina_test(aecjs 1 gen std seq)

nabla_add_okina_test(p1apwb1D_version_2 1 gen std seq)
nabla_add_okina_test(p1apwb1D_semi_implicite 1 gen std seq)
nabla_add_okina_test(p1apwb1D_two_state_nolinear 1 gen std seq)
nabla_add_okina_test(p1apwb1D_explicite 1 gen std seq)
nabla_add_okina_test(p1apwb1D_gosse 1 gen std seq)
nabla_add_okina_test(p1apwb1D_two_state_irregular_mesh 1 gen std seq)
nabla_add_okina_test(p1apwb1D_two_state_semi_implicte 1 gen std seq)

nabla_add_okina_test(p1apwb1D_version_2 1 run std seq)
nabla_add_okina_test(p1apwb1D_semi_implicite 1 run std seq)
nabla_add_okina_test(p1apwb1D_two_state_nolinear 1 run std seq)
nabla_add_okina_test(p1apwb1D_explicite 1 run std seq)
nabla_add_okina_test(p1apwb1D_gosse 1 run std seq)
nabla_add_okina_test(p1apwb1D_two_state_irregular_mesh 1 run std seq)
nabla_add_okina_test(p1apwb1D_two_state_semi_implicte 1 run std seq)

##########################
# NABLA TESTS FOR LAMBDA #
##########################
nabla_add_test(lambda glace2D 1 gen seq)
nabla_add_test(lambda gram 1 gen seq)
nabla_add_test(lambda lulesh 1 gen seq)
nabla_add_test(lambda lulesh 1 run seq)
nabla_add_test(lambda lulesh 1 run omp)

nabla_add_test(lambda llsh 1 gen seq)
nabla_add_test(lambda llsh 1 run seq)

nabla_add_test(lambda upwindAP 1 gen seq)
nabla_add_test(lambda upwindAP 1 run seq)

nabla_add_test(lambda heat 1 gen seq)
nabla_add_test(lambda aleph1D 1 gen seq)
nabla_add_test(lambda amber 1 gen seq)

nabla_add_test(lambda aleph2D 1 gen seq)

nabla_add_test(lambda hlt 1 gen seq)
nabla_add_test(lambda hlt 1 run seq)

if (HYPRE_FOUND)
   nabla_add_test(lambda aleph1D 1 run seq)
   nabla_add_test(lambda aleph2D 1 run seq)
   nabla_add_test(lambda heat 1 run seq)
   nabla_add_test(lambda amber 1 run seq)
   nabla_add_test_with_defines(lambda darcy 1 run seq -DALEPH_INDEX 100000)
endif (HYPRE_FOUND)

nabla_add_test_with_defines(lambda deflex 1 run seq -DNO_DEFINES 100000)

nabla_add_test(lambda p1apwb1D_version_2 1 gen std seq)
nabla_add_test(lambda p1apwb1D_semi_implicite 1 gen std seq)
nabla_add_test(lambda p1apwb1D_two_state_nolinear 1 gen std seq)
nabla_add_test(lambda p1apwb1D_explicite 1 gen std seq)
nabla_add_test(lambda p1apwb1D_gosse 1 gen std seq)
nabla_add_test(lambda p1apwb1D_implicite 1 gen std seq)
nabla_add_test(lambda p1apwb1D_two_state_irregular_mesh 1 gen std seq)
nabla_add_test(lambda p1apwb1D_two_state_semi_implicte 1 gen std seq)

nabla_add_test(lambda p1apwb1D_implicite 1 run std seq)


##########################
# NABLA TESTS FOR KOKKOS #
##########################
set(TESTS_NOT_YET_SUPPORTED_WITH_KOKKOS "shydro sethi anyItem gad comd pDDFV")
foreach(TST ${NABLA_TESTS})
  string(FIND ${TESTS_NOT_YET_SUPPORTED_WITH_KOKKOS} ${TST} TEST_SHOULD_PASS)
  if(${TEST_SHOULD_PASS} EQUAL -1)
    nabla_add_test(kokkos ${TST} 1 gen seq)
  endif()
endforeach(TST)
nabla_add_test(kokkos lulesh 1 run seq)
nabla_add_test(kokkos lulesh 1 run omp)
nabla_add_test(kokkos llsh 1 run seq)
nabla_add_test(kokkos upwindAP 1 run seq)
nabla_add_test(kokkos hlt 1 run seq)
nabla_add_test_with_defines(kokkos deflex 1 run seq -DNO_DEFINES 100000)


########################
# NABLA TESTS FOR CUDA #
########################
nabla_add_cuda_test(gram 1 gen)
nabla_add_cuda_test(nvknl 1 gen)
nabla_add_cuda_test(lulesh 1 gen)
nabla_add_cuda_test(mhydro 1 gen)
nabla_add_cuda_test(darcy 1 gen)
if(CUDA_FOUND)
  #nabla_add_cuda_test(mhydro 1 run)
  nabla_add_cuda_test(lulesh 1 run)
  nabla_add_cuda_test(nvknl 1 run)
  #nabla_add_cuda_test(darcy 1 run)
endif()
