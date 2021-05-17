
project( ${MIIND_LIBRARY_PREFIX}numtools )

link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
set( LIB_TYPE STATIC )

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
    ExStateDVIntegrator.h
    ExStateDVIntegratorCode.h
    GaussianDistribution.h
    GSLObjects.h
    IsApproximatelyEqualTo.h
    Interpolation.h
    LocalDefinitions.h
    MinMaxTracker.h
    MinMaxTrackerCode.h
    Modulo.h
    NumtoolsLib.h
    NumtoolsLibException.h
    PoissonDistribution.h
    Precision.h
    QaDirty.h
    QaDirtyImplementation.h
    RandomGenerator.h
    Rational.h
    TestDefinitions.h
    UniformDistribution.h )

include_directories( ${GSL_INCLUDE_DIRS} )
add_library( ${MIIND_LIBRARY_PREFIX}numtools ${LIB_TYPE} ${numtools_SRC} ${numtools_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS} )

install( FILES ${numtools_HEADERS}
    DESTINATION include/miind/NumtoolsLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}numtools
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
