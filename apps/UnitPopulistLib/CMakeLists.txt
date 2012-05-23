
if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module DynamicLibTest)." )
  endif( NOT GSL_FOUND )
    link_directories( ${GSL_LIBRARY_RELEASE_DIRS} )
else( WIN32 )
    link_directories( ${GSL_LIBRARY_DIRS} )
    include( RequireGSL )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )


include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

include_directories( ../../libs/DynamicLib )
include_directories( ../../libs/PopulistLib )


if ( Boost_FOUND )
    include_directories( ${Boost_INCLUDE_DIRS} )
    link_directories( ${Boost_LIBRARY_DIRS} )
    message( ${Boost_LIBRARY_DIRS} )
    if (UNIX)
        ADD_DEFINITIONS("-fno-strict-aliasing" )
    endif (UNIX)
    set ( LIBLIST 
        ${MIIND_LIBRARY_PREFIX}numtools
        ${MIIND_LIBRARY_PREFIX}util
        ${MIIND_LIBRARY_PREFIX}net
        ${MIIND_LIBRARY_PREFIX}sparseimplementation
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




add_executable( UnitPopulistLib
		UnitPopulistLib.cpp
		AbstractCirculantTest.cpp
		CharacteristicAlgorithmTest.cpp
		InhibitionTest.cpp
		InterpolationRebinnerTest.cpp
		NonCirculantSolverTest.cpp
		ProbabilityQueueTest.cpp
		QIFAlgorithmTest.cpp
		QIFParameterTest.cpp
		RefractiveCirculantSolverTest.cpp
		SingleInputZeroLeakEquationsTest.cpp
		 )		


target_link_libraries( UnitPopulistLib ${LIBLIST} )

