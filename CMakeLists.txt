execute_process(COMMAND "tput" "reset")
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
include(CMake/VT100.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# Welcome to the NABLA Project! #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")
project(Nabla)
string(FIND ${CMAKE_HOST_SYSTEM} "Linux-2.6" OLD_LINUX_KERNEL)
if (OLD_LINUX_KERNEL EQUAL 0)
  info("${VT100_BOLD}${VT100_FG_GREEN}Found an old 2.6 Linux kernel!${VT100_RESET}")
  set(OLD_LINUX_KERNEL 1)
  #find_library(LIBC_LIBRARY NAMES NAMES libc.a PATHS /usr/lib/x86_64-redhat-linux5E/lib64)
endif()
enable_language(ASM) 
set(CMAKE_C_FLAGS "-g -Wall -std=c99")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(NABLA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NABLA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#########
# APPLE #
#########
if(APPLE)
  message("APPLE=${APPLE}, setting ABI_UNDERSCORE")
  set(ABI_UNDERSCORE "_")
else(APPLE)
  unset(ABI_UNDERSCORE)
endif(APPLE)

###############
# FLEX, BISON #
###############
info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} subdirectory, looking for: ${VT100_FG_YELLOW}BISON${VT100_RESET}, ${VT100_FG_YELLOW}FLEX${VT100_RESET} and ${VT100_FG_YELLOW}CUDA${VT100_RESET}")
find_package(BISON)
info("\t${VT100_FG_MAGENTA}BISON${VT100_RESET} version is ${VT100_BOLD}${BISON_VERSION}${VT100_RESET}")
find_package(FLEX)
info("\t${VT100_FG_MAGENTA}FLEX${VT100_RESET} version is ${VT100_BOLD}${FLEX_VERSION}${VT100_RESET}")
if ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))
	info("${VT100_BOLD}${VT100_FG_RED}((NOT BISON_FOUND) OR (NOT FLEX_FOUND))${VT100_RESET} return")
	return()
endif ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))
   
bison_target(oParser orgopt.y ${CMAKE_CURRENT_BINARY_DIR}/orgopt.tab.c COMPILE_FLAGS "--report=all")
flex_target(oScanner orgopt.l ${CMAKE_CURRENT_BINARY_DIR}/orgopt.lex.c COMPILE_FLAGS "")
add_flex_bison_dependency(oScanner oParser)

bison_target(nParser nabla.y ${CMAKE_CURRENT_BINARY_DIR}/nabla.tab.c 
  COMPILE_FLAGS "--report=all")#--warnings=all,error --graph 
flex_target(nScanner nabla.l ${CMAKE_CURRENT_BINARY_DIR}/nabla.lex.c COMPILE_FLAGS "")
add_flex_bison_dependency(nScanner nParser)

########
# CUDA #
######## 
find_package(CUDA)
if (CUDA_FOUND)
  info("\t${VT100_FG_MAGENTA}CUDA${VT100_RESET} version is ${VT100_BOLD}${CUDA_VERSION_STRING}${VT100_RESET}")
endif()

############
# PACKAGES #
############
find_package(MPI)

include(CMake/FindARCANE.cmake)
if(ARCANE_FOUND)
  info("\t${VT100_FG_MAGENTA}ARCANE${VT100_RESET} has been found in ${VT100_BOLD}${ARCANE_ROOT}${VT100_RESET}")
endif()

include(CMake/FindHYPRE.cmake)
if(HYPRE_FOUND)
  info("\t${VT100_FG_MAGENTA}HYPRE${VT100_RESET} has been found in ${VT100_BOLD}${HYPRE_ROOT}${VT100_RESET}")
endif()

include(CMake/FindIceT.cmake)
if(ICET_FOUND)
  info("\t${VT100_FG_MAGENTA}IceT${VT100_RESET} has been found in ${VT100_BOLD}${ICET_ROOT}${VT100_RESET}")
  info("\t${VT100_FG_MAGENTA}IceT${VT100_RESET} libraries are ${VT100_BOLD}${ICET_LIBRARIES}${VT100_RESET}")
endif()
  
include(CMake/FindKokkos.cmake)
if(KOKKOS_FOUND)
  info("\t${VT100_FG_MAGENTA}KOKKOS${VT100_RESET} has been found in ${VT100_BOLD}${KOKKOS_ROOT}${VT100_RESET}")
  endif()
  
include(CMake/FindRAJA.cmake)
if(RAJA_FOUND)
  info("\t${VT100_FG_MAGENTA}RAJA${VT100_RESET} has been found in ${VT100_BOLD}${RAJA_ROOT}${VT100_RESET}")
endif()

################################
# Frontend, Middlend & Toolbox #
################################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/frontend)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/middlend)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/toolbox)

####################################
# Executable, librairie et install #
####################################
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBRARIES OFF)
if (NOT APPLE AND NOT OLD_LINUX_KERNEL)
   set(CMAKE_EXE_LINKER_FLAGS "-static")
endif()
configure_file(nabla.S ${CMAKE_CURRENT_BINARY_DIR}/nabla.asm.S)
add_executable(nabla nabla.c ${CMAKE_CURRENT_BINARY_DIR}/nabla.asm.S)
add_library(libNabla ${BISON_nParser_OUTPUTS} ${FLEX_nScanner_OUTPUTS})
add_executable(orgopt ${BISON_oParser_OUTPUTS} ${FLEX_oScanner_OUTPUTS})
target_link_libraries(orgopt NablaFrontend NablaToolbox)

############
# Backends #
############
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/arcane)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/okina)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/lambda)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/kokkos)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/raja)

#######################
# Backends common lib #
#######################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/lib)
if(HYPRE_FOUND)
    info("root ALEPH_LIB is ${VT100_FG_MAGENTA}${ALEPH_LIB}${VT100_RESET}")
    info("root ALEPH_HYPRE_LIB is ${VT100_FG_MAGENTA}${ALEPH_HYPRE_LIB}${VT100_RESET}")
endif()

##################
# Link libraries #
##################
target_link_libraries(nabla libNabla
                      NablaFrontend NablaToolbox
                      NablaMiddlend
                      NablaArcane NablaCuda NablaKokkos NablaRaja NablaLambda NablaOkina
                      NablaLibCall NablaLibHook NablaLibDump NablaLibAleph
                      NablaMiddlend)

###########
# INSTALL #
###########
set(CMAKE_BUILD_TYPE "Release")
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/nabla DESTINATION bin CONFIGURATIONS Release)

#########
# TESTS #
#########
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
