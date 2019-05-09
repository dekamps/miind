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
# UnitMPILib source and header files
#############################################################################

set (UNIT_MPI_SRC
    UnitMPILib.cpp
    DelayAssemblyTest.cpp
    HebbianAlgorithmTest.cpp
    ProbabilityQueueTest.cpp
    RateFunctorTest.cpp
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
    gslcblas
)


#############################################################################
# executables
#############################################################################
add_executable(UnitMPILib ${UNIT_MPI_SRC})
target_link_libraries(UnitMPILib ${LIBLIST})
