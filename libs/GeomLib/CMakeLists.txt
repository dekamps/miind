
project( ${MIIND_LIBRARY_PREFIX}geom )

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
AbstractNeuralDynamics.cpp
AbstractOdeSystem.cpp
BinEstimator.cpp
GeomInputConvertor.cpp
GeomParameter.cpp
LeakingOdeSystem.cpp
LifEstimator.cpp
LifNeuralDynamics.cpp
MasterFactory.cpp
MuSigmaScalarProduct.cpp
OdeParameter.cpp
OdeDtParameter.cpp
OUAlgorithm.cpp
QifOdeSystem.cpp
Response.cpp
SpikingNeuralDynamics.cpp
SpikingOdeSystem.cpp
SpikingQifNeuralDynamics.cpp
 )

set( GEOM_HEADERS
AbstractMasterEquation.hpp
AbstractNeuralDynamics.hpp
AbstractOdeSystem.hpp
BasicDefinitions.hpp
BinEstimator.hpp
CNZLCache.hpp
CNZLCacheCode.hpp
ConnectionSquaredProduct.hpp
CurrentCompensationParameter.hpp
CurrentParameter.hpp
DiffusionParameter.hpp
GeomInputConvertor.hpp
GeomAlgorithm.hpp
GeomAlgorithmCode.hpp
GeomInputConvertor.hpp
GeomLib.hpp
GeomLibException.hpp
GeomParameter.hpp
InitialDensityParameter.hpp  
InputParameterSet.hpp
GeomLibException.hpp
LeakingOdeSystem.hpp
LifEstimator.hpp
LifNeuralDynamics.hpp
LifParameter.hpp
MasterFactory.hpp
MasterParameter.hpp
MuSigma.hpp
MuSigmaScalarProduct.hpp
NeuronParameter.hpp
NumericalMasterEquation.hpp
NumericalMasterEquationCode.hpp
OdeParameter.hpp
OdeDtParameter.hpp
OUAlgorithm.hpp
QifOdeSystem.hpp
QifParameter.hpp
Response.hpp
ResponseParameter.hpp
SpikingLifNeuralDynamics.hpp
SpikingNeuralDynamics.hpp
SpikingOdeSystem.hpp
SpikingQifNeuralDynamics.hpp
)

add_library( ${MIIND_LIBRARY_PREFIX}geom ${LIB_TYPE} ${GEOM_SRC} ${GEOM_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}geom
    ${gsl_LINK_FLAGS} )


install( FILES ${GEOM_HEADERS}
    DESTINATION include/miind/GeomLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}geom
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
