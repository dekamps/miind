include_directories( ../../libs/MPILib )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )
include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST       
    ${MIIND_LIBRARY_PREFIX}mpi      
    ${MPI_LIBRARIES} 
    ${Boost_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}dynamic
    ${gsl_LINK_FLAGS} 
	gsl
)



add_executable( miind-mpi
    miind-mpi.cpp )



target_link_libraries( miind-mpi ${LIBLIST} )


