##############################################################################
# include external header files
##############################################################################

#include the MPILib build dir for the config file
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_BINARY_DIR}/../../libs/
)


include_directories( ../../libs/MPILib )
include_directories( ../../libs/GeomLib )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )
include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

#############################################################################
# UnitGeom source and header files
#############################################################################

set (UNIT_GEOM_SRC 
    BinEstimatorTest.cpp
    CurrentCompensationParameterTest.cpp							
    GeomInputConvertorTest.cpp		
    GeomParameterTest.cpp
    InitialDensityParameterTest.cpp	
    LeakingOdeSystemTest.cpp  
    LifNeuralDynamicsTest.cpp  															
    MuSigmaScalarProductTest.cpp  
    NeuronParameterTest.cpp	
    OdeParameterTest.cpp
    UnitGeomLib.cpp
)

#############################################################################
# Libraries
#############################################################################

if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST       
    ${MIIND_LIBRARY_PREFIX}geom
    ${MIIND_LIBRARY_PREFIX}mpi      
    ${MPI_LIBRARIES} 
    ${ROOT_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS} 
    gsl

)


#############################################################################
# executables
#############################################################################
add_executable(UnitGeomLib ${UNIT_GEOM_SRC})
target_link_libraries(UnitGeomLib ${LIBLIST})

