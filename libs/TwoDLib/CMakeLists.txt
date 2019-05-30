link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )
set( LIB_TYPE SHARED )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )


#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)

set( TWOD_SRC
Cell.cpp
CheckSystem.cpp
Mesh.cpp
CSRAdapter.cpp
MeshTree.cpp
CSRMatrix.cpp
Ode2DSystem.cpp
display.cpp
Ode2DSystemGroup.cpp
Euler.cpp
Fid.cpp
PolyGenerator.cpp
pugixml.cpp
QuadGenerator.cpp
Quadrilateral.cpp
Redistribution.cpp
Stat.cpp
kd.cpp
TransitionMatrix.cpp
LineSegment.cpp
TransitionMatrixGenerator.cpp
Master.cpp
MasterGrid.cpp
Triangle.cpp
TriangleGenerator.cpp
MasterGridJump.cpp
MasterGridSomaDendrite.cpp
Uniform.cpp
MasterOdeint.cpp
MasterOMP.cpp
)

set( TWOD_HEADERS
Bin.hpp
MasterOMP.hpp
MasterParameter.hpp
Cell.hpp
MeshAlgorithmCode.hpp
MeshAlgorithmCustomCode.hpp
CheckSystem.hpp
MeshAlgorithmCustom.hpp
MeshAlgorithm.hpp
Coordinates.hpp
Mesh.hpp
CSRAdapter.hpp
MeshTree.hpp
CSRMatrix.hpp
modulo.hpp
DensityAlgorithmInterface.hpp
display.hpp
Ode2DSystemGroup.hpp
Ode2DSystem.hpp
Euler.hpp
Point.hpp
Fid.hpp
PolyGenerator.hpp
FiducialElement.hpp
pugiconfig.hpp
GridAlgorithmCode.hpp
pugixml.hpp
GridAlgorithm.hpp
GridJumpAlgorithmCode.hpp
QuadGenerator.hpp
GridJumpAlgorithm.hpp
GridReport.hpp
Quadrilateral.hpp
GridSomaDendriteAlgorithmCode.hpp
GridSomaDendriteAlgorithm.hpp
Redistribution.hpp
InputCombination.hpp
Stat.hpp
TransitionList.hpp
kd.h
LineSegment.hpp
TransitionMatrixGenerator.hpp
TransitionMatrix.hpp
MasterGrid.hpp
TriangleGenerator.hpp
MasterGridJump.hpp
Triangle.hpp
TwoDLibException.hpp
MasterGridSomaDendrite.hpp
TwoDLib.hpp
Master.hpp
Uniform.hpp
MasterOdeint.hpp
XML.hpp
)

set( GL_HEADERS
include/GL/eglew.h
include/GL/freeglut.h
include/GL/freeglut_ext.h
include/GL/freeglut_std.h
include/GL/glew.h
include/GL/glut.h
include/GL/glxew.h
include/GL/wglew.h
)

#############################################################################
# Libraries
#############################################################################
if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST
    ${MIIND_LIBRARY_PREFIX}mpi
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS}
    gsl
    GL
    GLUT
)

add_library( ${MIIND_LIBRARY_PREFIX}twod ${LIB_TYPE} ${TWOD_SRC} ${TWOD_HEADERS} ${GL_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}twod ${LIBLIST})

install( FILES ${TWOD_HEADERS}
    DESTINATION include/miind/TwoDLib )

install( FILES ${GL_HEADERS}
    DESTINATION include/miind/TwoDLib/include/GL )

install( TARGETS ${MIIND_LIBRARY_PREFIX}twod
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
