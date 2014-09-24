#/usr/local/opendev1/gcc/cmake/2.8.6/doc/cmake-2.8/cmake.txt 
#execute_process(COMMAND "tput" "reset")

cmake_minimum_required(VERSION 3.0)

include(CMake/VT100.cmake)
info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# Welcome to the NABLA Project! #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")

project(Nabla)
enable_language(ASM) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(NABLA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#set(NABLA_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(NABLA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})


# FLEX, BISON
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} subdirectory, looking for ${VT100_FG_YELLOW}BISON${VT100_RESET}, ${VT100_FG_YELLOW}FLEX${VT100_RESET} and ${VT100_FG_YELLOW}CUDA")
find_package(BISON)
find_package(FLEX)
if ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))
	info("${VT100_BOLD}${VT100_FG_RED}((NOT BISON_FOUND) OR (NOT FLEX_FOUND))${VT100_RESET} return")
	return()
endif ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))
bison_target(nParser nabla.y ${CMAKE_CURRENT_BINARY_DIR}/nabla.tab.c COMPILE_FLAGS "--report=all --warnings=all,error")
flex_target(nScanner nabla.l ${CMAKE_CURRENT_BINARY_DIR}/nabla.lex.c)
add_flex_bison_dependency(nScanner nParser)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR opt/cuda/5.0)
find_package(CUDA)

# Nabla Frontend
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}FRONTEND${VT100_RESET}")
set(FRONTEND_C nabla.c nablaAstDfs.c nablaAstNode.c nablaAstTree.c nablaTimeTree.c nablaDebug.c nablaTools.c)

# Nabla MiddlEnd
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} gathering ${VT100_FG_YELLOW}MIDDLEND${VT100_RESET}")
set(MIDDLEND_C middlend/nablaMiddlend.c middlend/nablaItems.c middlend/nablaVariables.c 
               middlend/nablaOptions.c middlend/nablaModules.c middlend/nablaJobs.c middlend/nablaFunctions.c)

# BACKEND ARCANE
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}ARCANE${VT100_RESET}")
set(BACKEND_ARCANE_C backends/arcane/nccArc.c backends/arcane/nccArcConfig.c backends/arcane/nccArcModule.c
							backends/arcane/nccArcAxl.c backends/arcane/nccArcFunction.c 
                     backends/arcane/nccArcJob.c backends/arcane/nccArcJobCells.c
                     backends/arcane/nccArcJobNodes.c backends/arcane/nccArcJobFaces.c
                     backends/arcane/nccArcMain.c backends/arcane/nccArcHooks.c backends/arcane/nccArcSimd.c)
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}ARCANE${VT100_RESET} libraries:${VT100_RESET}")
info("\t${VT100_FG_YELLOW}GMP${VT100_RESET}")
info("\t${VT100_FG_YELLOW}DFT${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Mail${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Slurm${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Aleph${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Scheme${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Cartesian${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Materials${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Particles${VT100_RESET}")
info("\t${VT100_FG_YELLOW}Mathematica${VT100_RESET}")
set(LIBRARIES_ARCANE_C 	backends/arcane/libraries/nccArcLibAleph.c
								backends/arcane/libraries/nccArcLibScheme.c
								backends/arcane/libraries/nccArcLibCartesian.c 
								backends/arcane/libraries/nccArcLibMaterials.c
								backends/arcane/libraries/nccArcLibGmp.c
								backends/arcane/libraries/nccArcLibDft.c
								backends/arcane/libraries/nccArcLibParticles.c
								backends/arcane/libraries/nccArcLibMathematica.c 
								backends/arcane/libraries/nccArcLibSlurm.c
								backends/arcane/libraries/nccArcLibMail.c)

# BACKEND CUDA
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}CUDA${VT100_RESET}")
set(BACKEND_CUDA_C backends/cuda/nccCudaMain.c backends/cuda/nccCuda.c 
                      backends/cuda/nccCudaEnumerates.c backends/cuda/nccCudaVariables.c
                      backends/cuda/nccCudaMesh.c backends/cuda/nccCudaFunction.c
                      backends/cuda/nccCudaJob.c backends/cuda/nccCudaDump.S backends/cuda/nccCudaSimd.c)
# On spécifie en plus que nccCudaDump.S va utiliser incbin pour se compiler
set_property(SOURCE backends/cuda/nccCudaDump.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/outputs/*.*)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/outputs/real3.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/cuda/real3.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/outputs/debug.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/cuda/debug.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/outputs/error.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/cuda/error.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/outputs/extra.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/cuda/extra.h)


# BACKEND OKINA
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} adding ${VT100_FG_YELLOW}BACKEND${VT100_RESET}: ${VT100_BOLD}${VT100_FG_YELLOW}OKINA${VT100_RESET}")
set(BACKEND_OKINA_C 
    backends/okina/nOkina.c
    backends/okina/nOkinaSimdStd.c
    backends/okina/nOkinaSimdSse.c
    backends/okina/nOkinaSimdAvx.c
    backends/okina/nOkinaSimdMic.c
    backends/okina/nOkinaData.c
    backends/okina/nOkinaEnum.c
    backends/okina/nOkinaFunc.c
    backends/okina/nOkinaJobs.c
    backends/okina/nOkinaMain.c
    backends/okina/nOkinaMesh.c
    backends/okina/nOkinaParallelCilk.c
    backends/okina/nOkinaParallelOpenMP.c
    backends/okina/nOkinaParallelVoid.c
    backends/okina/nOkinaDump.S)
# On spécifie en plus que nOkinaDump.S va utiliser incbin pour se compiler
set_property(SOURCE backends/okina/nOkinaDump.S 
             PROPERTY OBJECT_DEPENDS 
             ${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/*.*)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knDbg.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knDbg.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knMsh.c ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knMsh.c)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knMth.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knMth.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knStdReal.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knStdReal.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knStdReal3.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knStdReal3.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knStdInteger.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knStdInteger.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knStdTernary.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knStdTernary.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knStdGather.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knStdGather.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knStdScatter.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knStdScatter.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knStdOStream.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knStdOStream.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knSseReal.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knSseReal.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knSseReal3.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knSseReal3.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knSseInteger.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knSseInteger.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knSseTernary.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knSseTernary.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knSseGather.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knSseGather.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knSseScatter.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knSseScatter.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knSseOStream.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knSseOStream.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knAvxReal.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knAvxReal.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knAvxReal3.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knAvxReal3.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knAvxInteger.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knAvxInteger.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knAvxTernary.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knAvxTernary.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knAvxGather.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knAvxGather.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knAvxScatter.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knAvxScatter.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knAvxOStream.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knAvxOStream.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knMicReal.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knMicReal.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knMicReal3.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knMicReal3.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knMicInteger.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knMicInteger.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knMicTernary.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knMicTernary.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knMicGather.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knMicGather.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knMicScatter.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knMicScatter.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina/outputs/knMicOStream.h ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nabla.dir/backends/okina/knMicOStream.h)


####################################
# Executable, librairie et install #
####################################
add_executable(nabla ${FRONTEND_C} ${MIDDLEND_C} 
    ${BACKEND_ARCANE_C} ${LIBRARIES_ARCANE_C}
    ${BACKEND_CUDA_C} 
    ${BACKEND_OKINA_C} 
    ${LIBRARIES_C} ${BISON_nParser_OUTPUTS} ${FLEX_nScanner_OUTPUTS})
target_link_libraries(nabla ${FLEX_LIBRARIES})
#install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/nabla DESTINATION bin)


#########
# TESTS #
#########
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)