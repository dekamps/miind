##############################################################################
# include the header files
##############################################################################

#one below that MPILib is included in the paths they are needed for the cmake config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)

include_directories(include)

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )
include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

##############################################################################
# add the source files
##############################################################################
set(MPI_SRCS

    src/AlgorithmGrid.cpp
    src/HebbianAlgorithm.cpp
    src/PersistantAlgorithm.cpp
    src/WilsonCowanAlgorithm.cpp
    # Add  boxcar algorithm here

    src/ProbabilityQueue.cpp
    
    src/report/handler/AbstractReportHandler.cpp
    src/report/handler/CanvasParameter.cpp
    src/report/handler/GraphKey.cpp
    src/report/handler/InactiveReportHandler.cpp
    src/report/handler/MinimalReportHandler.cpp
    src/report/handler/RootHighThroughputHandler.cpp
    src/report/handler/RootCanvasNoMPI.cpp
    src/report/handler/RootReportHandler.cpp
    src/report/handler/ValueHandlerHandler.cpp
    
    src/report/Report.cpp
    
    src/utilities/CircularDistribution.cpp
    src/utilities/Exception.cpp
    src/utilities/FileNameGenerator.cpp
    src/utilities/IterationNumberException.cpp
    src/utilities/Log.cpp
    src/utilities/MPIProxy.cpp
    src/utilities/ParallelException.cpp
    src/utilities/ProgressBar.cpp
    
    src/NetworkState.cpp
    src/SimulationRunParameter.cpp
)

#############################################################################
# Libraries
#############################################################################
if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST 
    ${Boost_LIBRARIES} 
    ${MPI_LIBRARIES} 
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS} 
	gsl    
)

add_library( ${MIIND_LIBRARY_PREFIX}mpi ${LIB_TYPE} ${MPI_SRCS} ${MPI_HEADER})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}mpi ${LIBLIST} )

#############################################################################
# Cmake generated header files
#############################################################################
#CONFIGURE_FILE(
#    ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.cmake 
#    ${CMAKE_CURRENT_BINARY_DIR}/config.hpp
#)


#############################################################################
# testing
#############################################################################

IF(ENABLE_TESTING AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
        ADD_SUBDIRECTORY(test)
ENDIF (ENABLE_TESTING AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))

install( FILES ${MPI_HEADERS}
    DESTINATION include/miind/MPILib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}mpi
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
