include_directories( ../../libs/MPILib )


if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST       
    ${MIIND_LIBRARY_PREFIX}mpi      
    ${MPI_LIBRARIES} 
    ${Boost_LIBRARIES}
)



add_executable( miind-mpi
    miind-mpi.cpp )



target_link_libraries( miind-mpi ${LIBLIST} )


