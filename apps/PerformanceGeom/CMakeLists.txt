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

set (ONELIF_GEOM_SRC 
	LifOne.cpp
)

set (ONELIF_CANVAS_GEOM_SRC 
	LifOneCanvas.cpp
)

set (TWOLIF_GEOM_SRC 
	LifTwo.cpp
)

set (ONEQIF_GEOM_SRC
        QifJumpPoisson.cpp)	

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
add_executable(LifOne ${ONELIF_GEOM_SRC})
target_link_libraries(LifOne ${LIBLIST})

add_executable(LifOneCanvas ${ONELIF_CANVAS_GEOM_SRC})
target_link_libraries(LifOneCanvas ${LIBLIST})

add_executable(LifTwo ${TWOLIF_GEOM_SRC})
target_link_libraries(LifTwo ${LIBLIST})

add_executable(QifOnePoisson ${ONEQIF_GEOM_SRC})
target_link_libraries(QifOnePoisson ${LIBLIST})


