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

set (UNIT_TWOD_SRC
2DUnit.cpp
AdapterTest.cpp
CSRTest.cpp
LineSegmentTest.cpp
Ode2DSystemTest.cpp
Ode2DSystemGroupTest.cpp
FixtureOde2DSystemGroup.cpp
TransitionMatrixTest.cpp
FidTest.cpp
MasterTest.cpp
QuadrilateralTest.cpp
TriangleTest.cpp
GeneratorTest.cpp
MeshAlgorithmTest.cpp
StatTest.cpp
UniformTest.cpp
KDTreeTest.cpp
MeshTest.cpp
TransitionMatrixGeneratorTest.cpp
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
    ${GLUT_LIBRARY}
)


#############################################################################
# Copy Test files to executable directory
#############################################################################

configure_file( aexp0cfa5d69-a740-4474-b8b6-b39870e2f5ef.fid ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( aexp0cfa5d69-a740-4474-b8b6-b39870e2f5ef.mesh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( aexpoverview.mesh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( condee2a5ff4-0087-4d69-bae3-c0a223d03693.mesh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( condee2a5ff4-0087-4d69-bae3-c0a223d03693.mat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( condee2a5ff4-0087-4d69-bae3-c0a223d03693.stat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( condee2a5ff4-0087-4d69-bae3-c0a223d03693.res ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( condee2a5ff4-0087-4d69-bae3-c0a223d03693.config ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( life21da341-bd4a-4077-93c2-4d11804d0882.model ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( life21da341-bd4a-4077-93c2-4d11804d0882_0.03_0_0_0_.mat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( simple.mesh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( test.stat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( multiple.stat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file( timefactor.mesh ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

#############################################################################
# executables
#############################################################################
add_executable(UnitTwoDLib ${UNIT_TWOD_SRC})
target_link_libraries(UnitTwoDLib ${LIBLIST})
