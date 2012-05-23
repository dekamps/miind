
project( ${MIIND_LIBRARY_PREFIX}sparseimplementation )

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module SparseImplementationLib)." )
  endif( NOT GSL_FOUND )
  set( LIB_TYPE STATIC )
else( WIN32 )
    include( RequireGSL )
    set( LIB_TYPE SHARED )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )

set( sparseimplementation_SRC
    SparseImplementationTest.cpp
    SparseLibTestOpenException.cpp
    SparseNodeCreationExample.cpp )

set( sparseimplementation_HEADERS
    AbstractSparseNode.h
    AbstractSparseNodeCode.h
    BasicDefinitions.h
    CollectSuccesorInformation.h
    CollectSuccessorInformationCode.h
    ConnectionProduct.h
    LayerWeightIterator.h
    LayerWeightIteratorCode.h
    LayerWeightIteratorThreshold.h
    LayerWeightIteratorThresholdCode.h
    LayeredSparseImplementation.h
    LayeredSparseImplementationCode.h
    LocalDefinitions.h
    NodeLinkCollectionFromSparseImplementation.h
    NodeLinkCollectionFromSparseImplementationCode.h
    ReverseImplementationException.h
    ReversibleLayeredSparseImplementation.h
    ReversibleLayeredSparseImplementationCode.h
    ReversibleSparseNode.h
    ReversibleSparseNodeCode.h
    SetIncrementedId.h
    SparseImplementation.h
    SparseImplementationCode.h
    SparseImplementationLib.h
    SparseImplementationTest.h
    SparseImplementationTestCode.h
    SparseLibException.h
    SparseLibImplementation.h
    SparseLibIteratorException.h
    SparseLibTestOpenException.h
    SparseNode.h
    SparseNodeCode.h
    SparseNodeCreationExample.h
    SparsePredecessorIterator.h
    SparsePredecessorIteratorCode.h
    TestDefinitions.h
    TestReverseImplementation.h
    TestReverseImplementationCode.h )


add_library( ${MIIND_LIBRARY_PREFIX}sparseimplementation ${LIB_TYPE} ${sparseimplementation_SRC} ${sparseimplementation_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}sparseimplementation
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}net
    ${gsl_LINK_FLAGS} )

install( FILES ${sparseimplementation_HEADERS}
    DESTINATION include/miind/SparseImplementationLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}sparseimplementation
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
