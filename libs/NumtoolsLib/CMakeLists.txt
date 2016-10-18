
project( ${MIIND_LIBRARY_PREFIX}numtools )

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module NumtoolsLib)." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_RELEASE_DIRS} )
  set(LIB_TYPE STATIC)
else( WIN32 )
  link_directories( ${GSL_LIBRARY_DIRS} )
  include( RequireGSL )
  set( LIB_TYPE SHARED )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )

set( numtools_SRC
    Distribution.cpp
    GaussianDistribution.cpp
    GSLObjects.cpp
    Interpolation.cpp
    PoissonDistribution.cpp
    RandomGenerator.cpp
    Rational.cpp
    UniformDistribution.cpp )

set( numtools_HEADERS
    AbstractDVIntegrator.h
    AbstractDVIntegratorCode.h
    BasicDefinitions.h
    ChangeSeedException.h
    Distribution.h
    DVIntegrator.h
    DVIntegratorCode.h
    DVIntegratorException.h
    DVIntegratorStateParameter.h
    exponentialdistribution.h
    GaussianDistribution.h
    GSLObjects.h
    IsApproximatelyEqualTo.h
    Interpolation.h
    LocalDefinitions.h
    MinMaxTracker.h
    MinMaxTrackerCode.h
    NumtoolsLib.h
    NumtoolsLibException.h
    PoissonDistribution.h
    QaDirty.h
    QaDirtyImplementation.h
    RandomGenerator.h
    Rational.h
    TestDefinitions.h
    UniformDistribution.h )

add_library( ${MIIND_LIBRARY_PREFIX}numtools ${LIB_TYPE} ${numtools_SRC} ${numtools_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS} )

install( FILES ${numtools_HEADERS}
    DESTINATION include/miind/NumtoolsLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}numtools
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
