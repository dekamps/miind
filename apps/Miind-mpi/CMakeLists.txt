if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_RELEASE_DIRS} )
else( WIN32 )
    include( RequireGSL )
    link_directories( ${GSL_LIBRARY_DIRS} )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )

find_package( Root )
if( NOT ROOT_INCLUDE_DIRS AND NOT ROOT_LIBRARY_DIRS )
    message( FATAL_ERROR "Root Library not found." )
endif( NOT ROOT_INCLUDE_DIRS AND NOT ROOT_LIBRARY_DIRS )

include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

include_directories( ../../libs/MPILib )


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
        ${MIIND_LIBRARY_PREFIX}mpi
        ${MIIND_LIBRARY_PREFIX}net
	    ${MIIND_LIBRARY_PREFIX}miind
        ${MIIND_LIBRARY_PREFIX}structnet
        ${MIIND_LIBRARY_PREFIX}dynamic
	    ${MIIND_LIBRARY_PREFIX}populist
        ${LIB_PREFIX}Core
        ${LIB_PREFIX}RIO
        ${LIB_PREFIX}Hist
        ${LIB_PREFIX}Tree
        ${LIB_PREFIX}Gpad
        ${gsl_LINK_FLAGS} 
        ${MPI_LIBRARIES} 
	gsl
	    ${CBLAS_PREFIX}cblas 
        ${Boost_LIBRARIES}
    )
else( Boost_FOUND)
    message( STATUS "Boost.TEST not found -- skipping the BOOST unit tests" )
endif ( Boost_FOUND )


add_executable( miind-mpi
    miind-mpi.cpp )



target_link_libraries( miind-mpi ${LIBLIST} )


