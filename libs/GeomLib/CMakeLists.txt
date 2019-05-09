
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
BinEstimator.hpp
CNZLCache.hpp
CNZLCacheCode.hpp
CurrentCompensationParameter.hpp
GeomInputConvertor.hpp
GeomAlgorithm.hpp
GeomAlgorithmCode.hpp
GeomLib.hpp
GeomParameter.hpp
InitialDensityParameter.hpp  
InputParameterSet.hpp
GeomLibException.hpp
LeakingOdeSystem.hpp
LifEstimator.hpp
LifNeuralDynamics.hpp
MasterFactory.hpp
MuSigma.hpp
MuSigmaScalarProduct.hpp
NeuronParameter.hpp
NumericalMasterEquation.hpp
NumericalMasterEquationCode.hpp
OdeParameter.hpp
OdeDtParameter.hpp
OUAlgorithm.hpp
QifOdeSystem.hpp
Response.hpp
ResponseParameter.hpp
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
