

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
    MPINetwork.cpp
    MPINode.cpp
    #add cpp files
     )

set( mpi_HEADERS
    MPINode.hpp
    MPINetwork.hpp
    BasicTypes.hpp
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
        ${MIIND_LIBRARY_PREFIX}clam
        ${MIIND_LIBRARY_PREFIX}numtools
        ${MIIND_LIBRARY_PREFIX}util
        ${MIIND_LIBRARY_PREFIX}net
        ${MIIND_LIBRARY_PREFIX}sparseimplementation
        ${LIB_PREFIX}Core
        ${LIB_PREFIX}RIO
        ${LIB_PREFIX}Hist
        ${LIB_PREFIX}Tree
        ${LIB_PREFIX}Table
		${LIB_PREFIX}Matrix
		${LIB_PREFIX}MathCore
		${LIB_PREFIX}MathMore
        ${LIB_PREFIX}Gpad
		${LIB_PREFIX}Graf
		${LIB_PREFIX}Postscript
        ${gsl_LINK_FLAGS}
        ${Boost_LIBRARIES} 
	    gsl
	    ${CBLAS_PREFIX}cblas
	    ${MPI_LIBRARIES} 
    )
else( Boost_FOUND)
    message( FATAL_ERROR "Boost.mpi not found -- cannot build without boost mpi" )
endif ( Boost_FOUND )

add_library( ${MIIND_LIBRARY_PREFIX}mpi ${LIB_TYPE} ${mpi_SRC} ${mpi_HEADERS} )


target_link_libraries( ${MIIND_LIBRARY_PREFIX}mpi ${LIBLIST} )



install( FILES ${mpi_HEADERS}
    DESTINATION include/miind/MPILib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}mpi
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
