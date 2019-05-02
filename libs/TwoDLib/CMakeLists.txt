
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
MasterOMP.cpp
Cell.cpp
CheckSystem.cpp
Mesh.cpp
CSRAdapter.cpp
MeshTree.cpp
Ode2DSystem.cpp
CSRMatrix.cpp
Ode2DSystemGroup.cpp
display.cpp
PolyGenerator.cpp
Euler.cpp
pugixml.cpp
Fid.cpp
QuadGenerator.cpp
Quadrilateral.cpp
Redistribution.cpp
Stat.cpp
TransitionMatrix.cpp
TransitionMatrixGenerator.cpp
kd.cpp
LineSegment.cpp
Triangle.cpp
TriangleGenerator.cpp
Master.cpp
MasterGrid.cpp
MasterGridJump.cpp
Uniform.cpp
MasterOdeint.cpp
)

set( TWOD_HEADERS
Bin.hpp
MasterOMP.hpp
Cell.hpp
MasterParameter.hpp
MeshAlgorithmCode.hpp
CheckSystem.hpp
MeshAlgorithm.hpp
Coordinates.hpp
Mesh.hpp
MeshTree.hpp
CSRAdapter.hpp
modulo.hpp
CSRMatrix.hpp
Ode2DSystemGroup.hpp
DensityAlgorithmInterface.hpp
Ode2DSystem.hpp
Point.hpp
display.hpp
PolyGenerator.hpp
pugiconfig.hpp
Euler.hpp
pugixml.hpp
Fid.hpp
QuadGenerator.hpp
FiducialElement.hpp
Quadrilateral.hpp
GridAlgorithmCode.hpp
GridAlgorithm.hpp
Redistribution.hpp
GridJumpAlgorithmCode.hpp
GridJumpAlgorithm.hpp
Stat.hpp
GridReport.hpp
TransitionList.hpp
InputCombination.hpp
TransitionMatrixGenerator.hpp
kd.h
TransitionMatrix.hpp
LineSegment.hpp
TriangleGenerator.hpp
Triangle.hpp
MasterGrid.hpp
TwoDLibException.hpp
TwoDLib.hpp
MasterGridJump.hpp
Master.hpp
Uniform.hpp
XML.hpp
MasterOdeint.hpp
)


add_library( ${MIIND_LIBRARY_PREFIX}twod ${LIB_TYPE} ${TWOD_SRC} ${TWOD_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}twod
    ${gsl_LINK_FLAGS} )

install( FILES ${TWOD_HEADERS}
    DESTINATION include/miind/TwoDLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}twod
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
