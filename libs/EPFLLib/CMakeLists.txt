
project( ${MIIND_LIBRARY_PREFIX}epfl )

link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )
set( LIB_TYPE STATIC )


#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)
				
set( EPFL_SRC
epfl.c
util.c
)

set( TWOD_HEADERS
epfl.h
)


add_library( ${MIIND_LIBRARY_PREFIX}epfl ${LIB_TYPE} ${EPFL_SRC} ${EPFL_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}epfl
    ${gsl_LINK_FLAGS} )

install( FILES ${TWOD_HEADERS}
    DESTINATION include/miind/EPFLLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}epfl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
