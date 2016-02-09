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

add_executable(MarPopExample population-example.cpp)
target_link_libraries(MarPopExample ${LIBLIST})
add_executable(MarWta1 wta1.cpp)
target_link_libraries(MarWta1 ${LIBLIST})
add_executable(MarWta2 wta2.cpp)
target_link_libraries(MarWta2 ${LIBLIST})
#add_executable(NVN_Blackboard blackboard-example.cpp)
#target_link_libraries(NVN_Blackboard ${LIBLIST})
add_executable(Circuit circuits.cpp)
target_link_libraries(Circuit ${LIBLIST})
