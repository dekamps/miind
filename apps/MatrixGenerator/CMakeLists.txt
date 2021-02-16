##############################################################################
# include the header files
##############################################################################

#include the MPILib build dir for the config file
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_BINARY_DIR}/../../libs/
)

include_directories( ../../libs/TwoDLib )
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
    ${MIIND_LIBRARY_PREFIX}twod
    ${MIIND_LIBRARY_PREFIX}geom
    ${MIIND_LIBRARY_PREFIX}mpi
    ${MPI_LIBRARIES}
    ${ROOT_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS}
    gsl
    ${GLUT_LIBRARY}
)


#############################################################################
# executables
#############################################################################
add_executable(MatrixGenerator ConstructResetMapping.cpp	CorrectStrays.cpp		GenerateMatrix.cpp  MatrixGenerator.cpp TranslationObject.cpp)
add_executable(Bind Bind.cpp Bind.hpp)
target_link_libraries(Bind            ${LIBLIST})
target_link_libraries(MatrixGenerator ${LIBLIST})

set(APP_BASE_DIR share/miind/apps/MatrixGenerator/)

install( TARGETS Bind DESTINATION ${APP_BASE_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install( TARGETS MatrixGenerator DESTINATION ${APP_BASE_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
