##############################################################################
# include the header files
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
# Libraries
#############################################################################

if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST       
    ${MIIND_LIBRARY_PREFIX}mpi           
    ${MIIND_LIBRARY_PREFIX}geom
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
#add_executable(twoPopulationDemo two-population.cpp)
#target_link_libraries(twoPopulationDemo ${LIBLIST})

#add_executable(wilsoncowanDemo miind-mpi.cpp)
#target_link_libraries(wilsoncowanDemo ${LIBLIST})

add_executable(populationDemo population-example.cpp)
target_link_libraries(populationDemo ${LIBLIST})
