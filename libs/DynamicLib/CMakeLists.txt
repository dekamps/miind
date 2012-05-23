ADD_SUBDIRECTORY( Swig )

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module DynamicLib)." )
  endif( NOT GSL_FOUND )
  set( LIB_TYPE STATIC )
else( WIN32 )
  include( RequireGSL )
  set( LIB_TYPE SHARED )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )

# not necessary any more as checked earlier
#find_package( Root )
#if( NOT ROOT_FOUND )
#    message( FATAL_ERROR "Root Library not found." )
#endif( NOT ROOT_FOUND )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

set( dynamic_SRC
    AbstractReportHandler.cpp
    AlgorithmBuilder.cpp
    AlgorithmGrid.cpp
    AsciiReportHandler.cpp
	CanvasParameter.cpp
    DynamicLibTest.cpp
    GraphKey.cpp
    InvariantReportHandler.cpp
    NetworkState.cpp
    NodeState.cpp
	RootFileInterpreter.cpp
	RootHighThroughputHandler.cpp
    RootReportHandler.cpp
    SimulationRunParameter.cpp
	UnitDynamic.cpp
    ValueHandlerHandler.cpp
    WilsonCowanAlgorithm.cpp
    WilsonCowanExample.cpp )

set( dynamic_HEADERS
    AbstractAlgorithm.h
    AbstractAlgorithmCode.h
    AbstractReportHandler.h
    AbstractSpecificAlgorithmParameter.h
    Afferent.h
    AfferentCode.h
    AlgorithmBuilder.h
    AlgorithmGrid.h
    AsciiReportHandler.h
    AsciiReportManager.h
    BasicDefinitions.h
    ConfigureNodes.h
	CanvasParameter.h
	DelayAlgorithm.h
	DelayAlgorithmCode.h
    DynamicLib.h
    DynamicLibException.h
    DynamicLibTest.h
    DynamicNetwork.h
    DynamicNetworkCode.h
    DynamicNetworkImplementation.h
    DynamicNetworkImplementationCode.h
    DynamicNode.h
    DynamicNodeCode.h
    EvolveNodes.h
    GraphKey.h
    InactiveReportHandler.h
    IterationNumberException.h
    LocalDefinitions.h
    NetworkState.h
    NodeInfo.h
    NodeState.h
    RateAlgorithm.h
    RateAlgorithmCode.h
    RateFunctor.h
    RateFunctorCode.h
    ReallocException.h
    Report.h
    ReportManager.h
    ReportManagerCode.h
	RootFileInterpreter.h
	RootHighThroughputHandler.h
    RootReportHandler.h
    RootReportHandlerPrivate.h
    SimulationRunParameter.h
    SpatialDynamicNetwork.h
    SpatialPosition.h
    StateConfigurationException.h
    TestDefinitions.h
    ValueHandlerHandler.h
    WeightedLink.h
    WilsonCowanAlgorithm.h
    WilsonCowanExample.h
    WilsonCowanParameter.h )

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
        ${MIIND_LIBRARY_PREFIX}sparseimplementation
        ${LIB_PREFIX}Core
        ${LIB_PREFIX}RIO
        ${LIB_PREFIX}Hist
        ${LIB_PREFIX}Tree
        ${LIB_PREFIX}Table
		${LIB_PREFIX}Matrix
		${LIB_PREFIX}MathCore
		${LIB_PREFIX}MathMore
        ${LIB_PREFIX}Gpad
		${LIB_PREFIX}Graf
		${LIB_PREFIX}Postscript
        ${gsl_LINK_FLAGS} 
	gsl
	${CBLAS_PREFIX}cblas 
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
else( Boost_FOUND)
    message( STATUS "Boost.TEST not found -- skipping the BOOST unit tests" )
endif ( Boost_FOUND )

add_library( ${MIIND_LIBRARY_PREFIX}dynamic ${LIB_TYPE} ${dynamic_SRC} ${dynamic_HEADERS} )


target_link_libraries( ${MIIND_LIBRARY_PREFIX}dynamic ${LIBLIST} )



install( FILES ${dynamic_HEADERS}
    DESTINATION include/miind/DynamicLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}dynamic
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
