##############################################################################
# include external header files
##############################################################################

#include the MPILib build dir for the config file
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_BINARY_DIR}/../../libs/
)


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

set (INT_TWOD_SRC 
)

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

)


#############################################################################
# Copy Test files to executable directory
#############################################################################

configure_file( condee2a5ff4-0087-4d69-bae3-c0a223d03693.model ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( aexp97b7e1e7-b88c-4ff8-8e86-7e1e092f4d9a.model ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( rinzel.model ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( rinzel_-0.1_0_0_0_.mat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( rinzel_0.1_0_0_0_.mat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( condee2a5ff4-0087-4d69-bae3-c0a223d03693.res ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( condee2a5ff4-0087-4d69-bae3-c0a223d03693.rev ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( condee2a5ff4-0087-4d69-bae3-c0a223d03693_0_0.05_0_0_.mat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( aexp97b7e1e7-b88c-4ff8-8e86-7e1e092f4d9a_-1_0_0_0_.mat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( aexp97b7e1e7-b88c-4ff8-8e86-7e1e092f4d9a_1_0_0_0_.mat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

#############################################################################
# executables
#############################################################################
add_executable(MeshAlgorithmGroupTest Euler.cpp MeshAlgorithmGroupTest.cpp)
target_link_libraries(MeshAlgorithmGroupTest ${LIBLIST})

add_executable(MasterGroupTest MasterGroupTest.cpp)
target_link_libraries(MasterGroupTest ${LIBLIST})

add_executable(cond cond.cpp)
target_link_libraries(cond ${LIBLIST})

add_executable(aexp  aexp_brette.cpp)
target_link_libraries(aexp ${LIBLIST})

add_executable(rinzel  Euler.cpp rinzel.cpp)
target_link_libraries(rinzel ${LIBLIST})


