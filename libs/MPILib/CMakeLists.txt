##############################################################################
# include the header files
##############################################################################

#one below that MPILib is included in the paths they are needed for the cmake config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)
if (WIN32)
set( LIB_TYPE STATIC )
else (WIN32)
set( LIB_TYPE SHARED )
endif (WIN32)

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
src/RefractoryQueue.cpp
src/DelayedConnectionQueueCode.cpp
src/HebbianAlgorithm.cpp
src/SimulationRunParameter.cpp
src/NetworkState.cpp
src/PersistantAlgorithm.cpp
src/WilsonCowanAlgorithm.cpp
src/ProbabilityQueue.cpp

src/report/Report.cpp

src/report/handler/AbstractReportHandler.cpp
src/report/handler/RootCanvasNoMPI.cpp
src/report/handler/CanvasParameter.cpp
src/report/handler/RootHighThroughputHandler.cpp
src/report/handler/GraphKey.cpp
src/report/handler/RootReportHandler.cpp
src/report/handler/InactiveReportHandler.cpp
src/report/handler/ValueHandlerHandler.cpp
src/report/handler/MinimalReportHandler.cpp

src/utilities/CircularDistribution.cpp
src/utilities/IterationNumberException.cpp
src/utilities/ParallelException.cpp
src/utilities/Exception.cpp
src/utilities/Log.cpp
src/utilities/ProgressBar.cpp
src/utilities/FileNameGenerator.cpp
src/utilities/MPIProxy.cpp
)

set(MPI_HEADER
include/AlgorithmGrid.hpp
include/NetworkState.hpp
include/AlgorithmInterface.hpp
include/NodeType.hpp
include/BasicDefinitions.hpp
include/PersistantAlgorithm.hpp
include/CustomConnectionParameters.hpp
include/DelayAlgorithmCode.hpp
include/ProbabilityQueue.hpp
include/DelayAlgorithm.hpp
include/RateAlgorithmCode.hpp
include/DelayAssemblyAlgorithmCode.hpp
include/RateAlgorithm.hpp
include/DelayAssemblyAlgorithm.hpp
include/RateFunctorCode.hpp
include/DelayAssemblyParameter.hpp
include/RateFunctor.hpp
include/DelayedConnection.hpp
include/RefractoryQueue.hpp
include/DelayedConnectionQueue.hpp
include/HebbianAlgorithm.hpp
include/SimulationRunParameter.hpp
include/HebbianParameter.hpp
include/StampedProbability.hpp
include/MiindTvbModelAbstract.hpp
include/StringDefinitions.hpp
include/MPINetworkCode.hpp
include/TypeDefinitions.hpp
include/MPINetwork.hpp
include/MPINodeCode.hpp
include/WilsonCowanAlgorithm.hpp
include/MPINode.hpp
include/WilsonCowanParameter.hpp
)

set(MPI_HEADER_POP
include/populist/RateFunctorCode.hpp
include/populist/RateFunctor.hpp
)

set(MPI_HEADER_REP
include/report/Report.hpp
include/report/ReportType.hpp
include/report/ReportValue.hpp
)

set(MPI_HEADER_REP_HANDL
include/report/handler/AbstractReportHandler.hpp
include/report/handler/CanvasParameter.hpp
include/report/handler/RootHighThroughputHandler.hpp
include/report/handler/GraphKey.hpp
include/report/handler/InactiveReportHandler.hpp
include/report/handler/RootReportHandler.hpp
include/report/handler/MinimalReportHandler.hpp
include/report/handler/RootCanvasMPI.hpp
include/report/handler/ValueHandlerHandler.hpp
include/report/handler/RootCanvasNoMPI.hpp
)

set(MPI_HEADER_UTIL
include/utilities/CircularDistribution.hpp
include/utilities/Log.hpp
include/utilities/ProgressBar.hpp
include/utilities/Exception.hpp
include/utilities/MPIProxy.hpp
include/utilities/Singleton.hpp
include/utilities/FileNameGenerator.hpp
include/utilities/NodeDistributionInterface.hpp
include/utilities/IterationNumberException.hpp
include/utilities/ParallelException.hpp
)

#############################################################################
# Libraries
#############################################################################
if (UNIX)
   ADD_DEFINITIONS("-fno-strict-aliasing" )
endif (UNIX)
set ( LIBLIST
    ${ROOT_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MPI_LIBRARIES}
    ${MIIND_LIBRARY_PREFIX}numtools
    ${gsl_LINK_FLAGS}
    gsl
)

add_library( ${MIIND_LIBRARY_PREFIX}mpi ${LIB_TYPE} ${MPI_SRCS} ${MPI_HEADER} ${MPI_HEADER_POP} ${MPI_HEADER_REP} ${MPI_HEADER_REP_HANDL} ${MPI_HEADER_UTIL})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}mpi ${LIBLIST})

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

install( FILES ${MPI_HEADER}
    DESTINATION include/miind/MPILib/include )
install( FILES ${MPI_HEADER_POP}
    DESTINATION include/miind/MPILib/include/populist )
install( FILES ${MPI_HEADER_REP}
    DESTINATION include/miind/MPILib/include/report )
install( FILES ${MPI_HEADER_REP_HANDL}
    DESTINATION include/miind/MPILib/include/report/handler )
install( FILES ${MPI_HEADER_UTIL}
    DESTINATION include/miind/MPILib/include/utilities )

install( TARGETS ${MIIND_LIBRARY_PREFIX}mpi
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
