
project( ${MIIND_LIBRARY_PREFIX}net )

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module NetLib)." )
  endif( NOT GSL_FOUND )
  set( LIB_TYPE STATIC )
else( WIN32 )
    include( RequireGSL )
    set( LIB_TYPE SHARED )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )


set( net_SRC
     AbstractArchitecture.cpp
     AbstractNodeLinkCollection.cpp
     AbstractSquashingFunction.cpp
     AbstractSquashingParameter.cpp
     Architecture.cpp
     DefaultLayerNodeLinkCollection.cpp
     DefaultNodeLinkCollection.cpp
     ImplementationLibTest.cpp
     LayeredArchitecture.cpp
     LayeredImplementation.cpp
     NetLibTest.cpp
     NodeId.cpp
     NodeLinkCollection.cpp
     NodeLink.cpp
     NoSquashingFunction.cpp
     NoSquashingParameter.cpp
     Sigmoid.cpp
     SigmoidParameter.cpp
     SquashingFunctionFactory.cpp
     WeightMatrixElementImplementation.cpp
)

set( net_HEADERS
     AbstractArchitecture.h
     AbstractNodeLinkCollection.h
     AbstractSquashingFunction.h
     AbstractSquashingParameter.h
     AdjacencyMatrixImplementation.h
     Architecture.h
     BasicDefinitions.h
     ConstNodeIteratorCode.h
     ConstNodeIterator.h
     DefaultLayerNodeLinkCollection.h
     DefaultNodeLinkCollection.h
     ImplementationLib.h
     ImplementationLibTest.h
     LayeredArchitecture.h
     LayeredImplementation.h
     LayerIteratorImplementation.h
     LayerStructure.h
     LayerVector.h
     LocalDefinitions.h
     NetLibException.h
     NetLib.h
     NetLibTestException.h
     NetLibTest.h
     NetworkParsingException.h
     NodeId.h
     NodeIteratorCode.h
     NodeIterator.h
     NodeLinkCollection.h
     NodeLinkCollectionPrivate.h
     NodeLink.h
     NoSquashingFunction.h
     NoSquashingParameter.h
     PatternCode.h
     Pattern.h
     PatternParsingException.h
     Sigmoid.h
     SigmoidParameter.h
     SquashingFunctionFactory.h
     Unimplemented.h
     UnknownSquashingException.h
     WeightMatrixElementImplementation.h
)


add_library( ${MIIND_LIBRARY_PREFIX}net ${LIB_TYPE} ${net_SRC} ${net_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}util
    ${gsl_LINK_FLAGS} )

install( FILES ${net_HEADERS}
    DESTINATION include/miind/NetLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}net
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
