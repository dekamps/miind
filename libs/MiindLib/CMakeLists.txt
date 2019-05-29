
project( ${MIIND_LIBRARY_PREFIX}shared )

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
VectorizedNetwork.cpp
)

set( TWOD_HEADERS
VectorizedNetwork.hpp
)

if (ENABLE_CUDA)
   if (NOT APPLE)
   list(APPEND TWOD_SRC
	VectorizedNetwork.cpp
   )
   list(APPEND TWOD_HEADERS
	VectorizedNetwork.hpp
   )
   endif (NOT APPLE)
endif(ENABLE_CUDA)

set(GLUT_LIBRARY "-lGL -lGLU -lglut")

if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST
${MIIND_LIBRARY_PREFIX}geom
${MIIND_LIBRARY_PREFIX}twod
${MIIND_LIBRARY_PREFIX}numtools
${MIIND_LIBRARY_PREFIX}mpi
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS}
	gsl
  ${GLUT_LIBRARY}
)

if (ENABLE_CUDA)
   if (NOT APPLE)
   list(APPEND LIBLIST ${MIIND_LIBRARY_PREFIX}cudatwod)
   endif (NOT APPLE)
endif(ENABLE_CUDA)

add_library( ${MIIND_LIBRARY_PREFIX}shared ${LIB_TYPE} ${TWOD_SRC} ${TWOD_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}shared ${LIBLIST})

install( FILES ${TWOD_HEADERS}
    DESTINATION include/miind/MiindLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
