##############################################################################
# include external header files
##############################################################################

#include the MPILib build dir for the config file
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_BINARY_DIR}/../../libs/
)


include_directories( ../../libs/MPILib )
include_directories( ../../libs/NumtoolsLib )
include_directories( ../../libs/TwoDLib )
include_directories( ../../libs/CudaTwoDLib )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )
include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

#############################################################################
# UnitGeom source and header files
#############################################################################

set (UNIT_CUDA_SRC
    CudaUnit.cpp
    AdapterTest.cpp
    FixtureAdapter.cpp
    MatrixTest.cpp
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
    ${MIIND_LIBRARY_PREFIX}cudatwod      
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

configure_file(condee2a5ff4-0087-4d69-bae3-c0a223d03693.model ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(condee2a5ff4-0087-4d69-bae3-c0a223d03693_0_0.05_0_0_.mat ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

#############################################################################
# executables
#############################################################################
add_executable(UnitCudaLib ${UNIT_CUDA_SRC})
target_link_libraries(UnitCudaLib ${LIBLIST})
add_executable(CudaEmulationTest CudaEmulationTest.cpp)
target_link_libraries(CudaEmulationTest ${LIBLIST})
