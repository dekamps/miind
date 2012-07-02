
##############################################################################
# add the source files
##############################################################################
set(MPI_SRCS
    src/algorithm/AlgorithmGrid.cpp
    src/algorithm/RateAlgorithm.cpp
    src/algorithm/WilsonCowanAlgorithm.cpp
    src/populist/AbstractCirculantSolver.cpp
    src/populist/AbstractRateComputation.cpp
    src/populist/AbstractRebinner.cpp
    src/populist/LIFConvertor.cpp
    src/populist/PopulistSpecificParameter.cpp
    src/report/handler/AbstractReportHandler.cpp
    src/report/handler/GraphKey.cpp
    src/report/handler/InactiveReportHandler.cpp
    src/report/handler/RootHighThroughputHandler.cpp
    src/report/handler/RootReportHandler.cpp
    src/report/handler/ValueHandlerHandler.cpp
    src/report/Report.cpp
    src/utilities/CircularDistribution.cpp
    src/utilities/Exception.cpp
    src/utilities/FileNameGenerator.cpp
    src/utilities/IterationNumberException.cpp
    src/utilities/LogStream.cpp
    src/utilities/ParallelException.cpp
    src/utilities/ProgressBar.cpp
    src/utilities/TimeException.cpp
    src/utilities/Timer.cpp
    src/NetworkState.cpp
    src/SimulationRunParameter.cpp
)

##############################################################################
# include the header files
##############################################################################
include_directories(include)
     
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
    ${gsl_LINK_FLAGS} 
	gsl
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
    
)

add_library( ${MIIND_LIBRARY_PREFIX}mpi ${LIB_TYPE} ${MPI_SRCS} ${MPI_HEADER})


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
