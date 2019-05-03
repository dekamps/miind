project( ${MIIND_LIBRARY_PREFIX}twod )

link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )
set( LIB_TYPE SHARED )


#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)

set( TWOD_SRC
CSRAdapter.cpp
CSRMatrix.cpp
display.cpp
Master.cpp
TransitionMatrix.cpp
Cell.cpp
MasterOMP.cpp
MasterOdeint.cpp
QuadGenerator.cpp
PolyGenerator.cpp
TransitionMatrixGenerator.cpp
pugixml.cpp
CheckSystem.cpp
Mesh.cpp
Quadrilateral.cpp
Triangle.cpp
kd.cpp
MasterGrid.cpp
MasterGridJump.cpp
Fid.cpp
Euler.cpp
MeshTree.cpp
Redistribution.cpp
TriangleGenerator.cpp
LineSegment.cpp
Ode2DSystem.cpp
Ode2DSystemGroup.cpp
Stat.cpp
Uniform.cpp
)

set( TWOD_HEADERS
display.hpp
GridAlgorithmCode.hpp
GridReport.hpp
Bin.hpp
InputCombination.hpp
MeshAlgorithmCode.hpp
Redistribution.hpp
TwoDLibException.hpp
CSRMatrix.hpp
LineSegment.hpp
MeshTree.hpp
Stat.hpp
TwoDLib.hpp
Euler.hpp
Cell.hpp
Master.hpp
MasterOdeint.hpp
MasterGrid.hpp
MasterGridJump.hpp
Ode2DSystem.hpp
Ode2DSystemGroup.hpp
TransitionList.hpp
Uniform.hpp
CheckSystem.hpp
MasterOMP.hpp
TransitionMatrix.hpp
modulo.hpp
Coordinates.hpp
MasterParameter.hpp
Point.hpp
TransitionMatrixGenerator.hpp
pugiconfig.hpp
Fid.hpp
Mesh.hpp
PolyGenerator.hpp
QuadGenerator.hpp
Triangle.hpp
pugixml.hpp
FiducialElement.hpp
MeshAlgorithm.hpp
Quadrilateral.hpp
TriangleGenerator.hpp
XML.hpp
)

#############################################################################
# Libraries
#############################################################################
if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS}
    gsl
)


add_library( ${MIIND_LIBRARY_PREFIX}twod ${LIB_TYPE} ${TWOD_SRC} ${TWOD_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}twod ${LIBLIST}
    ${gsl_LINK_FLAGS} ${MIIND_LIBRARY_PREFIX}mpi GL glut )

install( FILES ${numtools_HEADERS}
    DESTINATION include/miind/TwoDLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}twod
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
