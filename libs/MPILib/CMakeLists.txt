
set( mpi_SRC
    src/utilities/Exception.cpp
    src/utilities/ParallelException.cpp
    src/utilities/CircularDistribution.cpp
    src/utilities/Time.cpp
    src/WilsonCowanAlgorithm.cpp
    #add cpp files
     )

set( mpi_HEADERS
    include/MPINode.hpp
    include/MPINodeCode.hpp
    include/MPINetwork.hpp
    include/MPINetworkCode.hpp
    include/BasicTypes.hpp
    include/utilities/Exception.hpp
    include/utilities/ParallelException.hpp
    include/utilities/CircularDistribution.hpp
    include/utilities/NodeDistributionInterface.hpp
    include/AlgorithmInterface.hpp
    include/Sleep10secAlgorithm.hpp
    include/Sleep10secAlgorithmCode.hpp
    include/utilities/Time.hpp
    include/utilities/walltime.hpp
    include/WilsonCowanAlgorithm.hpp
    #add headers
     )
     
include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )
include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST 
    ${Boost_LIBRARIES} 
    ${MPI_LIBRARIES} 
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}dynamic
    ${gsl_LINK_FLAGS} 
	gsl
    
)

add_library( ${MIIND_LIBRARY_PREFIX}mpi ${LIB_TYPE} ${mpi_SRC} ${mpi_HEADERS})


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
