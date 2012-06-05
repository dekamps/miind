
set( mpi_SRC
    src/MPINetwork.cpp
    src/MPINode.cpp
    src/utilities/Exception.cpp
    src/utilities/ParallelException.cpp
    src/utilities/CircularDistribution.cpp
    src/EmptyAlgorithm.cpp
    src/utilities/Time.cpp
    #add cpp files
     )

set( mpi_HEADERS
    include/MPINode.hpp
    include/MPINetwork.hpp
    include/BasicTypes.hpp
    include/utilities/Exception.hpp
    include/utilities/ParallelException.hpp
    include/utilities/CircularDistribution.hpp
    include/utilities/NodeDistributionInterface.hpp
    include/AlgorithmInterface.hpp
    include/EmptyAlgorithm.hpp
    include/utilities/Time.hpp
    include/utilities/walltime.hpp
    #add headers
     )

if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST 
    ${Boost_LIBRARIES} 
    ${MPI_LIBRARIES} 
)

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
