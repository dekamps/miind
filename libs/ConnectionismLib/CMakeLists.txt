ADD_SUBDIRECTORY( Swig )

project( ${MIIND_LIBRARY_PREFIX}connectionism )


if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module ConnectionismLib)." )
  endif( NOT GSL_FOUND )
  set( LIB_TYPE STATIC )
else( WIN32 )
  include( RequireGSL )
  set( LIB_TYPE SHARED )  
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )


set( connectionism_SRC
    ConnectionismTest.cpp
    RandomOrder.cpp
    TrainingAlgorithm.cpp
    TrainingParameter.cpp
)

set( connectionism_HEADERS
    BackpropTraining.h
    BackpropTrainingCode.h
    BackpropTrainingVector.h
    BackpropTrainingVectorCode.h
    BasicDefinitions.h
    Connection.h
    ConnectionismLib.h
    ConnectionismLibException.h
    ConnectionismTest.h
    ConnectionismTestParameters.h
    Hebbian.h
    HebbianCode.h
    InconsistentIteratorException.h
    LayeredNetwork.h
    LayeredNetworkCode.h
    LocalDefinitions.h
    RandomOrder.h
    SquashingTestFunction.h
    TrainingAlgorithm.h
    TrainingParameter.h
    TrainingUnit.h
    TrainingUnitCode.h )

add_library( ${MIIND_LIBRARY_PREFIX}connectionism ${LIB_TYPE} ${connectionism_SRC} ${connectionism_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}connectionism
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}sparseimplementation
    ${MIIND_LIBRARY_PREFIX}net
    ${MIIND_LIBRARY_PREFIX}util
    gsl
    ${CBLAS_PREFIX}cblas )


install( FILES ${connectionism_HEADERS}
    DESTINATION include/miind/ConnectionismLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}connectionism
    LIBRARY DESTINATION lib 
    ARCHIVE DESTINATION lib )
