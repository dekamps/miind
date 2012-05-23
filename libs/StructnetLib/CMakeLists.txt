
project( ${MIIND_LIBRARY_PREFIX}structnet )


if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module structnet)." )
  endif( NOT GSL_FOUND )
  set(LIB_TYPE STATIC)
  link_directories( ${GSL_LIBRARY_RELEASE_DIRS} )
else( WIN32 )
  include( RequireGSL )
  link_directories( ${GSL_LIBRARY_DIRS} )
  set(LIB_TYPE SHARED)
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )

set( structnet_SRC
    ActivityRect.cpp
    ActivityRectangleAttributes.cpp
    DenseOverlapLinkRelation.cpp
    DenseOverlapColorLinkRelation.cpp
    ForwardOrder.cpp
    FullyConnectedLinkRelation.cpp
    LayerDescription.cpp
    LinkRelation.cpp
    NodeIdPosition.cpp
    NoLinkRelation.cpp
    PhysicalPosition.cpp
    ReverseDenseOverlapLinkRelation.cpp
    RZOrder.cpp
    StructnetLibTest.cpp )

set( structnet_HEADERS
    ActivityRect.h
    ActivityRectangleAttributes.h
    BasicDefinitions.h
    DenseOverlapColorLinkRelation.h
    DenseOverlapLinkRelation.h
    DescriptionException.h
    FullyConnectedLinkRelation.h
    ForwardOrder.h
    InconsistentStructureException.h
    LayerDescription.h
    LinkRelation.h
    LocalDefinitions.h
    NeuronIdPositionParsingException.h
    NodeIdPosition.h
    NoLinkRelation.h
    OrientedPattern.h
    OrientedPatternCode.h
    PhysicalPosition.h
    ReverseOrder.h
    ReverseDenseOverlapLinkRelation.h
    RZOrder.h
    SparseLayeredNet.h
    SparseLayeredNetCode.h
    StructnetLib.h
    StructnetLibException.h
    StructnetLibTest.h
    XORLinkRelation.h )

add_library( ${MIIND_LIBRARY_PREFIX}structnet ${LIB_TYPE} ${structnet_SRC} ${structnet_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}structnet
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}connectionism
    ${MIIND_LIBRARY_PREFIX}sparseimplementation
    ${gsl_LINK_FLAGS} )

install( FILES ${structnet_HEADERS}
    DESTINATION include/miind/StructnetLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}structnet
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
