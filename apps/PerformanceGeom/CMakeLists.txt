##############################################################################
# include external header files
##############################################################################

#include the MPILib build dir for the config file
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_BINARY_DIR}/../../libs/
)

find_package(gsl REQUIRED)
include_directories( ../../libs/MPILib )
include_directories( ../../libs/GeomLib )
include_directories( ../../libs/TwoDLib )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )
include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

#############################################################################
# UnitGeom source and header files
#############################################################################

set (ONEPOP_TWOD_SRC 
	OnePop.cpp
)

set (ONELIF_GEOM_SRC 
	LifOne.cpp
)

set (ONELIF_CANVAS_GEOM_SRC 
	LifOneCanvas.cpp
)

set (TWOLIF_GEOM_SRC 
	LifTwo.cpp
)

set (TWOLIF_CANVAS_GEOM_SRC 
	LifTwoCanvas.cpp
)

set (ONEQIF_GEOM_SRC
        QifJumpPoisson.cpp)	

set (QIF_GEOM_SRC_CORRELATED
        QifCorrelated.cpp)	

set (RESPONSE_DATA
        ResponseData.cpp)	

#############################################################################
# Libraries
#############################################################################

if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST       
    ${MIIND_LIBRARY_PREFIX}geom
    ${MIIND_LIBRARY_PREFIX}twod      
    ${MIIND_LIBRARY_PREFIX}mpi      
    ${MPI_LIBRARIES} 
    ${ROOT_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS} 
    gsl
    gslcblas
    ${GLUT_LIBRARY}
)


#############################################################################
# executables
#############################################################################
add_executable(OnePop ${ONEPOP_TWOD_SRC})
target_link_libraries(OnePop ${LIBLIST})

add_executable(LifOne ${ONELIF_GEOM_SRC})
target_link_libraries(LifOne ${LIBLIST})

add_executable(LifOneCanvas ${ONELIF_CANVAS_GEOM_SRC})
target_link_libraries(LifOneCanvas ${LIBLIST})

add_executable(LifTwo ${TWOLIF_GEOM_SRC})
target_link_libraries(LifTwo ${LIBLIST})

add_executable(LifTwoCanvas ${TWOLIF_CANVAS_GEOM_SRC})
target_link_libraries(LifTwoCanvas ${LIBLIST})

add_executable(QifJumpPoisson ${ONEQIF_GEOM_SRC})
target_link_libraries(QifJumpPoisson ${LIBLIST})

add_executable(QifCorrelated ${QIF_GEOM_SRC_CORRELATED})
target_link_libraries(QifCorrelated ${LIBLIST})

add_executable(ResponseData ${RESPONSE_DATA})
target_link_libraries(ResponseData ${LIBLIST})


