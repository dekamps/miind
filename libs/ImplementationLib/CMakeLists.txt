
project( ${MIIND_LIBRARY_PREFIX}implementation )

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module ImplementationLib)." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_DIRS} )
  set( LIB_TYPE STATIC )
else( WIN32 )
  include( RequireGSL )
  set( LIB_TYPE SHARED )
endif( WIN32 )

set( implementation_SRC
    AbstractSquashingFunction.cpp
    AbstractSquashingParameter.cpp
    ImplementationLibTest.cpp
    LayeredImplementation.cpp
    NoSquashingFunction.cpp
    NoSquashingParameter.cpp
    Sigmoid.cpp
    SigmoidParameter.cpp
    SquashingFunctionFactory.cpp
    WeightMatrixElementImplementation.cpp )

set( implementation_HEADERS
    AbstractSquashingFunction.h
    AbstractSquashingParameter.h
    AdjacencyMatrixImplementation.h
    BasicDefinitions.h
    ConstNodeIterator.h
    ConstNodeIteratorCode.h
    ImplementationLib.h
    ImplementationLibTest.h
    LayeredImplementation.h
    LocalDefinitions.h
    NodeIterator.h
    NodeIteratorCode.h
    NoSquashingFunction.h
    NoSquashingParameter.h
    Sigmoid.h
    SigmoidParameter.h
    SquashingFunctionFactory.h
    UnknownSquashingException.h
    WeightMatrixElementImplementation.h )

add_library( ${MIIND_LIBRARY_PREFIX}implementation ${LIB_TYPE} ${implementation_SRC} ${implementation_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}implementation
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}util
    ${gsl_LINK_FLAGS} )

install( FILES ${implementation_HEADERS}
    DESTINATION include/miind/ImplementationLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}implementation
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
