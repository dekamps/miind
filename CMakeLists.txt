project( miind )
cmake_minimum_required(VERSION 2.6)
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules )

# include cmake scripts
#############################################################################
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules )

# require out-of-source build
INCLUDE(require_out_of_src_build)
MACRO_REQUIRE_OUT_OF_SRC_BUILD("Miind requires out-of-source builds.")

# default install prefix. can be set with ccmake
set( CMAKE_INSTALL_PREFIX "/usr/local" CACHE STRING "Install prefix." )


set( MIIND_LIBRARY_PREFIX "miind" CACHE STRING "prefix string for all libs" )

##############################################################################
# compile flags
##############################################################################
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wextra -Wno-unused-parameter -Wno-error=unused-parameter -O0 -pedantic")

#for cross compiling remove the -march=native flag!
if (APPLE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -fPIC -std=c++11")
else (APPLE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -fext-numeric-literals -fPIC")
endif(APPLE)
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")


##############################################################################
# cmake options
##############################################################################

OPTION(ENABLE_OPENMP "Enable OpenMP" ON)
OPTION(ENABLE_CUDA "Enable CUDA" OFF)
OPTION(ENABLE_TESTING "Compile tests" OFF)
OPTION(ENABLE_ROOT "Root Desired" OFF)
OPTION(ENABLE_MEMORY_TEST "Generate memory test" OFF)
OPTION(ENABLE_MPI "Enable mpi" OFF)
OPTION(ENABLE_COVERAGE "Enable GCov coverage analysis (defines a 'coverage' target and enforces static build of miind)" OFF)

set(DEBUGLEVEL "logINFO" CACHE STRING "Define the Debug Level")
add_definitions( -DDEBUGLEVEL=${DEBUGLEVEL})

##############################################################################
# enable MPI dependent build
##############################################################################

# Require MPI for this project:
IF(ENABLE_MPI)
  MESSAGE(STATUS "MPI enabled")
	find_package(MPI REQUIRED)
	add_definitions( -DENABLE_MPI=1 )
ENDIF(ENABLE_MPI)

IF(ENABLE_OPENMP)
	find_package(OpenMP REQUIRED)
	add_definitions( -DENABLE_OPENMP=1 )
ENDIF(ENABLE_OPENMP)



IF(ENABLE_OPENMP)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF(ENABLE_OPENMP)

set(GLUT_LIBRARY "-lGL -lGLU -lglut")

IF(ENABLE_MPI)
	set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
	set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
	include_directories(${MPI_INCLUDE_PATH})
ENDIF(ENABLE_MPI)
##############################################################################
# find boost with mpi
##############################################################################
IF(ENABLE_MPI)
	find_package( Boost COMPONENTS unit_test_framework filesystem program_options mpi thread serialization timer system python)
ELSE(ENABLE_MPI)
	find_package( Boost COMPONENTS unit_test_framework filesystem program_options thread serialization timer system python)
ENDIF(ENABLE_MPI)
if( Boost_FOUND )
   message(STATUS "Boost found setting up directories")
   message(STATUS "BOOST_ROOT: " ${BOOST_ROOT} )
   message(STATUS "Boost include directories:" ${Boost_INCLUDE_DIR} )
   message(STATUS "Boost library directories:" ${Boost_LIBRARY_DIRS} )
   include_directories( ${Boost_INCLUDE_DIR} )
   link_directories( ${Boost_LIBRARY_DIR} )
else ( Boost_FOUND )
   message(FATAL_ERROR "Boost is mandatory to build the libraries")
endif( Boost_FOUND )
##############################################################################
#
##############################################################################
IF(ENABLE_ROOT)
find_package( Root )
if( ROOT_FOUND )
  message(STATUS "Building Root dependent parts")
  message(STATUS "Root include directories:" ${ROOT_INCLUDE_DIRS})
  message(STATUS "Root library directories:" ${ROOT_LIBRARY_DIRS})
  set (ROOT_LIBRARIES Core Hist Tree Table Matrix MathCore MathMore Gpad Graf Postscript RIO)
  add_definitions(-DWANTROOT)
else( ROOT_FOUND )
  message(FATAL_ERROR "Root is required for MPILib")
endif( ROOT_FOUND )

# TODO:  This is ugly and must be fixed in the windows install procedure
if (WIN32)
SET (LIB_PREFIX lib)
else (WIN32)
SET (CBLAS_PREFIX gsl)
endif (WIN32)
ENDIF (ENABLE_ROOT)
#############################################################################
# code coverage analysis
#############################################################################

IF (ENABLE_COVERAGE)
	IF(CMAKE_BUILD_TYPE STREQUAL "Debug" AND ENABLE_TESTING)
		MESSAGE(STATUS "Coverage enabled")
		MESSAGE(WARNING "At the moment lcov does not work with gcc 4.7, therefore coverage fails")
		INCLUDE(CodeCoverage)
		SETUP_TARGET_FOR_COVERAGE(coverage test coverageReport)
	ELSEIF(NOT ENABLE_TESTING)
		MESSAGE(FATAL_ERROR "Coverage analysis requires Testing Enabled")
	ELSE()
		MESSAGE(FATAL_ERROR "Coverage analysis requires DEBUG build")
	ENDIF()
ELSE()
   MESSAGE(STATUS "Coverage disabled")
ENDIF()


#############################################################################
# documentation
#############################################################################
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/doc/Doxyfile.cmake
${CMAKE_BINARY_DIR}/doc/Doxyfile
@ONLY IMMEDIATE
)

FIND_PACKAGE(Doxygen)

# target for doxygen
ADD_CUSTOM_TARGET(doc COMMAND ${DOXYGEN} ${CMAKE_BINARY_DIR}/doc/Doxyfile)

IF(ENABLE_TESTING AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
	MESSAGE(STATUS "Testing enabled")
	INCLUDE(CTest)
	ENABLE_TESTING()
ELSE(ENABLE_TESTING AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
	MESSAGE(STATUS "Testing disabled, enable testing and Debug build")
ENDIF(ENABLE_TESTING AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))



include_directories( ./libs )
message("========================================Common settings done going on with libs===================================")
ADD_SUBDIRECTORY( libs )
message("========================================libs set and done continuing with apps====================================")
ADD_SUBDIRECTORY( apps )
message("========================================compiler flags====================================")
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})


ADD_SUBDIRECTORY( python )
ADD_SUBDIRECTORY( examples )

