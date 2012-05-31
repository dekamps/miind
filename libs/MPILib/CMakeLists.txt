

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module DynamicLib)." )
  endif( NOT GSL_FOUND )
  set( LIB_TYPE STATIC )
else( WIN32 )
  include( RequireGSL )
  set( LIB_TYPE SHARED )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

set( mpi_SRC
    src/MPINetwork.cpp
    src/MPINode.cpp
    src/utilities/Exception.cpp
    src/utilities/ParallelException.cpp
    #add cpp files
     )

set( mpi_HEADERS
    include/MPINode.hpp
    include/MPINetwork.hpp
    include/BasicTypes.hpp
    include/utilities/Exception.hpp
    include/utilities/ParallelException.hpp
    #add headers
     )

if ( Boost_FOUND )
    include_directories( ${Boost_INCLUDE_DIRS} )
    link_directories( ${Boost_LIBRARY_DIRS} )
    message( ${Boost_LIBRARY_DIRS} )
    if (UNIX)
        ADD_DEFINITIONS("-fno-strict-aliasing" )
    endif (UNIX)
    set ( LIBLIST 
        ${Boost_LIBRARIES} 
	    ${MPI_LIBRARIES} 
    )
else( Boost_FOUND)
    message( FATAL_ERROR "Boost.mpi not found -- cannot build without boost mpi" )
endif ( Boost_FOUND )

add_library( ${MIIND_LIBRARY_PREFIX}mpi ${LIB_TYPE} ${mpi_SRC} ${mpi_HEADERS} )


target_link_libraries( ${MIIND_LIBRARY_PREFIX}mpi ${LIBLIST} )


#############################################################################
# tests
#############################################################################

IF (ENABLE_TESTING)
  ADD_SUBDIRECTORY(test)
ENDIF (ENABLE_TESTING)

install( FILES ${mpi_HEADERS}
    DESTINATION include/miind/MPILib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}mpi
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
