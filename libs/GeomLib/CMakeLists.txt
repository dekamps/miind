
project( ${MIIND_LIBRARY_PREFIX}numtools )

link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )
set( LIB_TYPE STATIC )


#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)
				
set( GEOM_SRC
AbstractOdeSystem.cpp
AbstractNeuralDynamics.cpp
CurrentCompensationParameter.hpp
GeomInputConvertor.cpp
GeomParameter.cpp
LeakingOdeSystem.cpp
LifNeuralDynamics.cpp
MuSigmaScalarProduct.cpp
OdeParameter.cpp
 )

set( GEOM_HEADERS
AbstractNeuralDynamics.hpp
AbstractOdeSystem.hpp
GeomInputConvertor.hpp
GeomLib.hpp
GeomParameter.hpp
InitialDensityParameter.hpp  
InputParameterSet.hpp
LeakingOdeSystem.hpp
LifNeuralDynamics.hpp
MuSigma.hpp
MuSigmaScalarProduct.hpp
OdeParameter.hpp
OrnsteinUhlenbeckParameter.hpp
 )

add_library( ${MIIND_LIBRARY_PREFIX}geom ${LIB_TYPE} ${GEOM_SRC} ${GEOM_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}geom
    ${MIIND_LIBRARY_PREFIX}util
    ${gsl_LINK_FLAGS} )

install( FILES ${numtools_HEADERS}
    DESTINATION include/miind/GeomLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}geom
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
