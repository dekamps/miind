project( miind )

cmake_minimum_required(VERSION 2.6)
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# include cmake scripts
#############################################################################
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules )

# require out-of-source build
INCLUDE(require_out_of_src_build)
MACRO_REQUIRE_OUT_OF_SRC_BUILD("Miind requires out-of-source builds.")

# default install prefix. can be set with ccmake
set( CMAKE_INSTALL_PREFIX "/usr/local" CACHE STRING "Install prefix." )

set( MIIND_LIBRARY_PREFIX "miind" CACHE STRING "prefix string for all libs" )

# On linux, we define all libs as shared so they can be used in
# generated python import libraries. On windows, we don't need to do that.
# If we need dlls in windows instead of static lib files, 
# we need to set DLL export macro here and will need to go through *ALL*
# classes we want to export... yeesh...
#if (WIN32)
#add_compile_definitions("DllExport=__declspec(dllexport)")
#else(WIN32)
#add_compile_definitions("DllExport=")
#endif(WIN32)

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
# compile flags
##############################################################################
if(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wextra -Wno-unused-parameter -Wno-error=unused-parameter -O0 -pedantic")
endif(CMAKE_COMPILER_IS_GNUCXX)

if (APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
	
	if(CMAKE_COMPILER_IS_GNUCXX)
		set(GLUT_LIBRARY "-framework OpenGL -framework GLUT")
	else()
		set(GLUT_LIBRARY "-lGL -lGLU -lGLUT")
	endif()
else (APPLE)
	if(CMAKE_COMPILER_IS_GNUCXX)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals -fPIC")
		set(GLUT_LIBRARY "-lGL -lGLU -lglut")
	endif(CMAKE_COMPILER_IS_GNUCXX)
endif(APPLE)
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")

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
	if (APPLE)
		if(ENABLE_OPENMP)
			if(CMAKE_C_COMPILER_ID MATCHES "Clang")
			  message ( "Setting OpenMP_C_FLAGS for Clang.")
			  set(OpenMP_C "${CMAKE_C_COMPILER}")
			  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -Wno-unused-command-line-argument")
			  set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
			  set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
		      set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
			  set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
			endif()
			if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
			  message ( "Setting OpenMP_CXX_FLAGS for Clang.")
			  set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
			  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -Wno-unused-command-line-argument")
			  set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
			  set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
			  set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
			  set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
			endif()
		endif()
	endif()
	
	find_package(OpenMP REQUIRED)
	
	if (OpenMP_FOUND AND APPLE)
		message ( "Found OpenMP. Try to find omp.h..." )
		find_path(OpenMP_INCLUDE_DIR
			omp.h
		)
		
		message ( "Adding include directories: ${OpenMP_INCLUDE_DIR} ")
		include_directories("${OpenMP_INCLUDE_DIR}")
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	endif(OpenMP_FOUND AND APPLE)
	
	add_definitions( -DENABLE_OPENMP=1 )
ENDIF(ENABLE_OPENMP)

IF (WIN32) # For windows packaging, we want to include the glut dll so we need FindGLUT to find it during install
find_package(GLUT REQUIRED)
ENDIF (WIN32)

IF(ENABLE_CUDA)
set(CMAKE_CUDA_ARCHITECTURES "sm_60") #adapt this to the GPU on your system
ENDIF(ENABLE_CUDA)


IF(ENABLE_OPENMP)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF(ENABLE_OPENMP)



IF(ENABLE_MPI)
	set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
	set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
	include_directories(${MPI_INCLUDE_PATH})
ENDIF(ENABLE_MPI)
##############################################################################
# find boost with mpi
##############################################################################
IF(ENABLE_MPI)
	find_package( Boost COMPONENTS unit_test_framework filesystem program_options mpi thread serialization timer system)
ELSE(ENABLE_MPI)
	find_package( Boost COMPONENTS unit_test_framework filesystem program_options thread serialization timer system)
ENDIF(ENABLE_MPI)
if( Boost_FOUND )
   message(STATUS "Boost found setting up directories")
   message(STATUS "BOOST_ROOT: " ${BOOST_ROOT} )
   message(STATUS "Boost include directories:" ${Boost_INCLUDE_DIR} )
   message(STATUS "Boost library directories:" ${Boost_LIBRARY_DIRS} )
   include_directories( ${Boost_INCLUDE_DIR} )
   link_directories( ${Boost_LIBRARY_DIRS} )
else ( Boost_FOUND )
   message("Boost is not found. Non MPI only.")
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
  set (ROOT_LIBRARIES Core Hist Tree Matrix MathCore MathMore Gpad Graf Postscript RIO)
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

if (NOT SKBUILD) # If miind is being built as a standalone (not a pip wheel)
message("======================================== Please update ENV variables ====================================")
message("")
message("export PATH=$PATH:${CMAKE_INSTALL_PREFIX}/share/miind/python/miind")
message("export PYTHONPATH=$PYTHONPATH:${CMAKE_INSTALL_PREFIX}/share/miind/python/")
message("export OMP_NUM_THREADS=<NUMBER OF AVAILABLE CORES>" ) 
message("")
message("==========================================================================================================")
endif (NOT SKBUILD)

