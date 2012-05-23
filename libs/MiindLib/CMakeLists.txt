project( ${MIIND_LIBRARY_PREFIX}miind )

if( WIN32)
  set( LIB_TYPE STATIC )
else( WIN32 )
  include( RequireGSL )
  set( LIB_TYPE SHARED )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )

find_package( Root )
if( NOT ROOT_FOUND )
    message( FATAL_ERROR "Root Library not found. Is essential for MiindLib." )
endif( NOT ROOT_FOUND )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

set( miind_SRC
	 MiindTest.cpp
	 SimulationBuilder.cpp
	 XMLNodes.cpp
	 XMLRunParameter.cpp
)

set(miind_HEADERS
	 MiindLib.h
	 MiindTest.h
     Simulation.h
     SimulationCode.h
	 SimulationBuilder.h
	 SimulationBuilderCode.h
     SimulationParser.h
     SimulationParserCode.h
	 XMLConnection.h
	 XMLConnectionCode.h
     XMLSimulation.h
	 XMLSimulationCode.h
	 XMLRunParameter.h
     XMLNodes.h
)

add_library( ${MIIND_LIBRARY_PREFIX}miind ${LIB_TYPE} ${miind_SRC} ${miind_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}miind
    ${MIIND_LIBRARY_PREFIX}numtools
	${MIIND_LIBRARY_PREFIX}clam
    ${MIIND_LIBRARY_PREFIX}util
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}connectionism
    ${MIIND_LIBRARY_PREFIX}structnet
    ${MIIND_LIBRARY_PREFIX}populist
    ${LIB_PREFIX}Tree
    gsl
    ${CBLAS_PREFIX}cblas )
    
if ( NOT CPPUNIT_FOUND )
#    find_package( CPPUNIT )
endif ( NOT CPPUNIT_FOUND )

install( FILES ${dynamic_HEADERS}
    DESTINATION include/miind/MiindLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}miind
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
