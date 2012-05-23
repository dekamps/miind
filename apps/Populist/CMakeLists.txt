
if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module PopulistTest)." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_DEBUG_DIRS} )
  add_definitions( -DGSL_DLL=1)
else( WIN32 )
    include( RequireGSL )
    link_directories( ${GSL_LIBRARY_DIRS} )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )

find_package( Root )
if( NOT ROOT_FOUND )
    message( FATAL_ERROR "Root Library not found. Is essential for ClamLib." )
endif( NOT ROOT_FOUND )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )


include_directories( ../../libs/PopulistLib )

add_executable( populistTest
    PopulistTest.cpp )

add_executable( populistExample
    PopulistExample.cpp )
add_executable( ExperimentResponseCurve
    ExperimentResponseCurve.cpp )

add_executable( WorkFlow WorkFlow.cpp)

add_executable( Analyze Analyze.cpp)


if ( Boost_FOUND )
    include_directories( ${Boost_INCLUDE_DIRS} )
    link_directories( ${Boost_LIBRARY_DIRS} )
    message( ${Boost_LIBRARY_DIRS} )
    if (UNIX)
        ADD_DEFINITIONS("-fno-strict-aliasing" )
    endif (UNIX)
    set ( LIBLIST 
        ${MIIND_LIBRARY_PREFIX}clam
        ${MIIND_LIBRARY_PREFIX}numtools
        ${MIIND_LIBRARY_PREFIX}util
        ${MIIND_LIBRARY_PREFIX}net
        ${MIIND_LIBRARY_PREFIX}net
	${MIIND_LIBRARY_PREFIX}miind
        ${MIIND_LIBRARY_PREFIX}structnet
        ${MIIND_LIBRARY_PREFIX}dynamic
	${MIIND_LIBRARY_PREFIX}populist
        ${LIB_PREFIX}Core
        ${LIB_PREFIX}RIO
        ${LIB_PREFIX}Hist
        ${LIB_PREFIX}Tree
        ${LIB_PREFIX}Gpad
        ${gsl_LINK_FLAGS} 
	gsl
	${CBLAS_PREFIX}cblas 
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
else( Boost_FOUND)
    message( STATUS "Boost.TEST not found -- skipping the BOOST unit tests" )
endif ( Boost_FOUND )


target_link_libraries( ExperimentResponseCurve ${LIBLIST} )

target_link_libraries( populistTest ${LIBLIST} )

target_link_libraries( populistExample ${LIBLIST} )


target_link_libraries( WorkFlow ${LIBLIST} )

target_link_libraries( Analyze ${LIBLIST} )

