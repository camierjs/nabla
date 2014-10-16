#/usr/local/opendev1/gcc/cmake/2.8.6/doc/cmake-2.8/cmake.txt 
execute_process(COMMAND "tput" "reset")

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#set(ENV{MAKEFLAGS} "--no-print-directory")
#set(CMAKE_MAKE_PROGRAM "gmake" CACHE FILEPATH "Allow to alias gmake to add --no-print-directory")
#set_property(CACHE CMAKE_MAKE_PROGRAM PROPERTY VALUE "gmake")

include(CMake/VT100.cmake)
info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# Welcome to the NABLA Project! #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")

project(Nabla)
enable_language(ASM) 
set(CMAKE_C_FLAGS "-Wall")#-std=c99
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(NABLA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NABLA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})


###############
# FLEX, BISON #
###############
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} subdirectory, looking for ${VT100_FG_YELLOW}BISON${VT100_RESET}, ${VT100_FG_YELLOW}FLEX${VT100_RESET} and ${VT100_FG_YELLOW}CUDA")
find_package(BISON)
find_package(FLEX)
if ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))
	info("${VT100_BOLD}${VT100_FG_RED}((NOT BISON_FOUND) OR (NOT FLEX_FOUND))${VT100_RESET} return")
	return()
endif ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))
bison_target(nParser nabla.y ${CMAKE_CURRENT_BINARY_DIR}/nabla.tab.c COMPILE_FLAGS "--report=all --warnings=all,error")#--graph 
flex_target(nScanner nabla.l ${CMAKE_CURRENT_BINARY_DIR}/nabla.lex.c)
add_flex_bison_dependency(nScanner nParser)


########
# CUDA #
########
set(CUDA_TOOLKIT_ROOT_DIR opt/cuda/5.0)
find_package(CUDA)


####################################
# Executable, librairie et install #
####################################
add_executable(nabla nabla.c)
add_library(Nabla ${BISON_nParser_OUTPUTS} ${FLEX_nScanner_OUTPUTS})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/frontend)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/middlend)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/arcane)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina)

# Still NablaMiddlend x2 because of HWed calls to backend entry-points
target_link_libraries(nabla Nabla ${FLEX_LIBRARIES} NablaMiddlend NablaArcane NablaCuda NablaOkina NablaMiddlend NablaFrontend)
#install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/nabla DESTINATION bin)

#########
# TESTS #
#########
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)