execute_process(COMMAND "tput" "reset")
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
include(CMake/VT100.cmake)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
info("${VT100_BOLD}CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}${VT100_RESET}")

info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# Welcome to the NABLA Project! #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")
project(Nabla)
enable_language(ASM) 
set(CMAKE_C_FLAGS "-Wall -std=c99")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(NABLA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NABLA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})


###############
# FLEX, BISON #
###############
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} subdirectory, looking for ${VT100_FG_YELLOW}BISON${VT100_RESET}, ${VT100_FG_YELLOW}FLEX${VT100_RESET} and ${VT100_FG_YELLOW}CUDA")
find_package(BISON)
info("\t${VT100_FG_MAGENTA}BISON${VT100_RESET} version is ${VT100_BOLD}${BISON_VERSION}${VT100_RESET}")
find_package(FLEX)
info("\t${VT100_FG_MAGENTA}FLEX${VT100_RESET} version is ${VT100_BOLD}${FLEX_VERSION}${VT100_RESET}")
if ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))
	info("${VT100_BOLD}${VT100_FG_RED}((NOT BISON_FOUND) OR (NOT FLEX_FOUND))${VT100_RESET} return")
	return()
endif ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))
bison_target(nParser nabla.y ${CMAKE_CURRENT_BINARY_DIR}/nabla.tab.c 
  COMPILE_FLAGS "--report=all --warnings=all,error")#--graph 
flex_target(nScanner nabla.l ${CMAKE_CURRENT_BINARY_DIR}/nabla.lex.c COMPILE_FLAGS "")
add_flex_bison_dependency(nScanner nParser)


########
# CUDA #
######## 
find_package(CUDA)
info("\t${VT100_FG_MAGENTA}CUDA${VT100_RESET} version is ${VT100_BOLD}${CUDA_VERSION_STRING}${VT100_RESET}")
#info("${VT100_FG_MAGENTA}CUDA${VT100_RESET} root dir is ${CUDA_TOOLKIT_ROOT_DIR}${VT100_RESET}")
#info("${VT100_FG_MAGENTA}CUDA${VT100_RESET} toolkit dir is ${CUDA_TOOLKIT_ROOT_DIR}${VT100_RESET}")
#info("${VT100_FG_MAGENTA}CUDA${VT100_RESET} includes are in ${CUDA_INCLUDE_DIRS}${VT100_RESET}")
#info("${VT100_FG_MAGENTA}CUDA${VT100_RESET} libs are ${CUDA_LIBRARIES}${VT100_RESET}")


##########
# ARCANE #
##########
include(CMake/FindARCANE.cmake)
find_package(MPI)
find_package(ARCANE)
info("\t${VT100_FG_MAGENTA}ARCANE${VT100_RESET} libs are in ${VT100_BOLD}${ARCANE_LIBRARY_DIR}${VT100_RESET}")
info("\t${VT100_FG_MAGENTA}ARCANE${VT100_RESET} includes are in ${VT100_BOLD}${ARCANE_INCLUDE_DIR}${VT100_RESET}")


#######################
# Frontend & Middlend #
#######################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/frontend)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/middlend)


####################################
# Executable, librairie et install #
####################################
add_executable(nabla nabla.c)
add_library(Nabla ${BISON_nParser_OUTPUTS} ${FLEX_nScanner_OUTPUTS})


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/arcane)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina)

# Still NablaMiddlend x2 because of HWed calls to backend entry-points
target_link_libraries(nabla Nabla ${FLEX_LIBRARIES} NablaMiddlend NablaArcane NablaCuda NablaOkina NablaMiddlend NablaFrontend)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/nabla DESTINATION bin)

#########
# TESTS #
#########
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)