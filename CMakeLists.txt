project( miind )
cmake_minimum_required(VERSION 2.6)
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules )
set( CMAKE_INSTALL_PREFIX "/usr/local" CACHE STRING "Install prefix." ) # default install prefix. can be set with ccmake


set( MIIND_LIBRARY_PREFIX "miind" CACHE STRING "prefix string for all libs" )




#
# Options
#

OPTION(ENABLE_TESTING "Compile tests" ON)
OPTION(ENABLE_COVERAGE "Enable GCov coverage analysis (defines a 'coverage' target and enforces static build of libpipe)" OFF)
OPTION(ENABLE_MPI "Enable threading" FALSE)

IF(ENABLE_MPI)
    MESSAGE(STATUS "MPI enabled")
    # Require MPI for this project:
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
    include_directories(${MPI_INCLUDE_PATH})
    #does not work with macport need to be teste for linux
ELSE()
    MESSAGE(STATUS "MPI disabled")   
ENDIF()
IF(ENABLE_TESTING)
    MESSAGE(STATUS "Testing enabled")
    INCLUDE(CTest)
    ENABLE_TESTING()
ELSE()
    MESSAGE(STATUS "Testing disabled")
ENDIF()


if (WIN32)
        ADD_DEFINITIONS( -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -DBOOST_PROGRAM_OPTIONS_NO_LIB -DBOOST_PROGRAM_OPTIONS_DYN_LINK)
else (WIN32)
	ADD_DEFINITIONS( -Wall ) # Extra compile options, just under Unix.  This would set /W4 underwindows
endif (WIN32)

# This used to be done with CMAKE_CXX_FLAG, however this is a  crude method overriding sensible default settings on some platforms
SET(SWIG_COMPILE_OPTIONS "-fno-strict-aliasing" ) # gnu optimising is too agressive for SWIG produced code according to SWIG documentation

#check for third party libraries
find_package( PkgConfig )

MARK_AS_ADVANCED( USE_FFTWD )
OPTION( USE_FFTWD "Use double precision FFTW if found" ON )
MARK_AS_ADVANCED( USE_FFTWF )
OPTION( USE_FFTWF "Use single precision FFTW if found" ON )

find_package( FFTW )
find_package( Boost COMPONENTS unit_test_framework program_options mpi serialization)
if( Boost_FOUND )
   message("Boost found setting up directories")
   message("BOOST_ROOT: " ${BOOST_ROOT} )
   message("Boost include directories:" ${Boost_INCLUDE_DIR} )
   message("Boost library directories:" ${Boost_LIBRARY_DIRS} )
   include_directories( ${Boost_INCLUDE_DIR} )
   link_directories( ${Boost_LIBRARY_DIR} )
else ( Boost_FOUND )
   message("Boost is mandatory to build the libraries")
endif( Boost_FOUND )

pkg_check_modules( GTKMM gtkmm-2.4 )
pkg_check_modules(IMAGEMAGICKLIB ImageMagick)
pkg_check_modules(IMAGEMAGICKLIB++ ImageMagick++)
pkg_check_modules(IMAGEMAGICKLIBWAND Wand)
find_package( Root )
if( ROOT_FOUND )
  message("Building Root dependent parts")
else( ROOT_FOUND )
  message("Building parts not depending on Root")
endif( ROOT_FOUND )

set(AGENT OFF)

# TODO:  This is ugly and must be fixed in the windows install procedure
if (WIN32)
SET (LIB_PREFIX lib)
else (WIN32)
SET (CBLAS_PREFIX gsl)
endif (WIN32)


#----------------swig
FIND_PACKAGE( SWIG )

if (SWIG_FOUND)
	INCLUDE( ${SWIG_USE_FILE} )
	FIND_PACKAGE( PythonLibs )
	INCLUDE_DIRECTORIES( ${PYTHON_INCLUDE_PATH} )
endif (SWIG_FOUND)

SET( CMAKE_SWIG_FLAGS "" )
#---------swig




include_directories( ./libs )
message("========================================Common settings done going on with libs===================================")
ADD_SUBDIRECTORY( libs )
message("========================================libs set and done continuing with apps====================================")
ADD_SUBDIRECTORY( apps )
