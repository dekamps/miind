link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )

set( LIB_TYPE SHARED )

if (WIN32) # When cmake generates the VS files, it needs to know this is a cuda project
enable_language(CUDA)
endif (WIN32)

find_package(CUDA  REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS} )

find_package (Python3 COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )


#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
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
    ${MIIND_LIBRARY_PREFIX}twod
    ${gsl_LINK_FLAGS}
    gsl
    ${GLUT_LIBRARY}
)

if (ENABLE_CUDA)
   if (NOT APPLE)
   list(APPEND LIBLIST ${MIIND_LIBRARY_PREFIX}cudatwod ${MIIND_LIBRARY_PREFIX}shared)
   endif (NOT APPLE)
endif(ENABLE_CUDA)

add_library( ${MIIND_LIBRARY_PREFIX}simv ${LIB_TYPE} MiindPython.cpp)

target_link_libraries( ${MIIND_LIBRARY_PREFIX}simv ${LIBLIST})

if (WIN32) # for windows, we want to set the ending to .pyd. In linux, the ending is just .so
SET_TARGET_PROPERTIES(${MIIND_LIBRARY_PREFIX}simv   PROPERTIES SUFFIX .pyd)
endif (WIN32)

install( TARGETS ${MIIND_LIBRARY_PREFIX}simv)



