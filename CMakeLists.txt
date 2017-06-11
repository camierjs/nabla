cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

######################################
# Test if we're not in top directory #
######################################
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "Cannot build in top directory,\
                       please (mkdir build && cd build && cmake ..)")
endif(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})

#####################
# Clean and process #
#####################
execute_process(COMMAND "tput" "reset")
include(cmake/VT100.cmake)

################
# Nabla banner #
################
info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}# Welcome to the NABLA Project! #${VT100_RESET}")
info("${VT100_BOLD}${VT100_FG_GREEN}#################################${VT100_RESET}")

#################
# Nabla project #
#################
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
project(Nabla C CXX ASM)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_VERBOSE_MAKEFILE YES)
set(CMAKE_INSTALL_PREFIX /usr/local/nabla)
#set(CMAKE_INSTALL_DO_STRIP "1")

#####################################
# Test if we're an old Linux kernel #
#####################################
string(FIND ${CMAKE_HOST_SYSTEM} "Linux-2.6" OLD_LINUX_KERNEL)
if (OLD_LINUX_KERNEL EQUAL 0)
  info("${VT100_BOLD}${VT100_FG_GREEN}Found an old 2.6 Linux kernel!${VT100_RESET}")
  set(OLD_LINUX_KERNEL 1)
  #find_library(LIBC_LIBRARY NAMES NAMES libc.a PATHS /usr/lib/x86_64-redhat-linux5E/lib64)
endif(OLD_LINUX_KERNEL EQUAL 0)

####################
# FLAGS & INCLUDES #
####################
set(CMAKE_C_FLAGS "-g -Wall -std=c99")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nabla)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/nabla)
set(NABLA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NABLA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

####################################
# Executable, librairie et install #
####################################
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBRARIES OFF)
if (NOT APPLE AND NOT OLD_LINUX_KERNEL)
   set(CMAKE_EXE_LINKER_FLAGS "-static")
endif(NOT APPLE AND NOT OLD_LINUX_KERNEL)

#############
# APPLE ABI #
#############
if(APPLE)
  message("APPLE=${APPLE}, setting ABI_UNDERSCORE")
  set(ABI_UNDERSCORE "_")
else(APPLE)
  unset(ABI_UNDERSCORE)
endif(APPLE)

###############
# FLEX, BISON #
###############
#info("${VT100_BOLD}${VT100_FG_CYAN}NABLA${VT100_RESET} subdirectory, looking for: ${VT100_FG_YELLOW}BISON${VT100_RESET}, ${VT100_FG_YELLOW}FLEX${VT100_RESET} and ${VT100_FG_YELLOW}CUDA${VT100_RESET}")
find_package(BISON REQUIRED)
info("\t${VT100_FG_MAGENTA}BISON${VT100_RESET} version is ${VT100_BOLD}${BISON_VERSION}${VT100_RESET}")
find_package(FLEX REQUIRED)
info("\t${VT100_FG_MAGENTA}FLEX${VT100_RESET} version is ${VT100_BOLD}${FLEX_VERSION}${VT100_RESET}")
if ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))
	message(FATAL_ERROR "${VT100_BOLD}${VT100_FG_RED}((NOT BISON_FOUND) OR (NOT FLEX_FOUND))${VT100_RESET}")
endif ((NOT BISON_FOUND) OR (NOT FLEX_FOUND))

############
# PACKAGES #
############
include(${NABLA_SOURCE_DIR}/cmake/CMakeTPL.txt)
find_package(MPI)
find_package(CUDA)
find_package(OpenMP)
include(cmake/FindARCANE.cmake)
include(cmake/FindHYPRE.cmake)
include(cmake/FindIceT.cmake)
include(cmake/FindKokkos.cmake)
include(cmake/FindRAJA.cmake)
include(cmake/FindPETSc.cmake)
include(cmake/FindMFEM.cmake)
include(cmake/FindLegion.cmake)

##################
# SUBDIRECTORIES #
##################
add_subdirectory(nabla)
add_subdirectory(orgopt)
add_subdirectory(frontend)
add_subdirectory(middlend)
add_subdirectory(tools)
add_subdirectory(aleph)
add_subdirectory(glvis)
add_subdirectory(okina)

add_subdirectory(backends/arcane)
add_subdirectory(backends/cuda)
add_subdirectory(backends/okina)
add_subdirectory(backends/lambda)
add_subdirectory(backends/kokkos)
add_subdirectory(backends/legion)
add_subdirectory(backends/raja)
add_subdirectory(backends/lib)

#########
# TESTS #
#########
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)


#########
# Emacs #
#########
install(DIRECTORY etc
        DESTINATION .
        DIRECTORY_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)
