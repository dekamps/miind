project( MiindRunGpu )

link_directories( ${GSL_LIBRARY_DIRS} )

#if (WIN32) # When cmake generates the VS files, it needs to know this is a cuda project
#enable_language(CUDA)
#endif (WIN32)

find_package(CUDA  REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS} )


link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )

include_directories( ${GSL_INCLUDE_DIRS} )
include_directories( ../../../libs/ )
include_directories( ../../../libs/GeomLib )
include_directories( ../../../libs/TwoDLib )

set(ENABLE_MPI OFF)

#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)

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

if (ENABLE_CUDA)
   if (NOT APPLE)
   list(APPEND LIBLIST ${MIIND_LIBRARY_PREFIX}cudatwod ${MIIND_LIBRARY_PREFIX}shared)
   endif (NOT APPLE)
endif(ENABLE_CUDA)

add_executable(MiindRunGpu MiindRunGPU.cpp)
target_link_libraries(MiindRunGpu  ${LIBLIST})

set(APP_BASE_DIR share/miind/apps/)

install( TARGETS MiindRunGpu DESTINATION ${APP_BASE_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
