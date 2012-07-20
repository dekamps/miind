##############################################################################
# include the header files
##############################################################################

#one below that MPILib is included in the paths they are needed for the camke config files
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
    src/algorithm/AlgorithmGrid.cpp
    src/algorithm/WilsonCowanAlgorithm.cpp
    
    src/populist/parameters/PopulistSpecificParameter.cpp
    src/populist/parameters/AdaptationParameter.cpp
    src/populist/parameters/OneDMParameter.cpp
    src/populist/zeroLeakEquations/LIFConvertor.cpp
    src/populist/zeroLeakEquations/LIFZeroLeakEquations.cpp
    src/populist/zeroLeakEquations/NumericalZeroLeakEquations.cpp
    src/populist/zeroLeakEquations/OldLifZeroLeakEquations.cpp
    src/populist/zeroLeakEquations/OneDMZeroLeakEquations.cpp
    src/populist/zeroLeakEquations/SingleInputZeroLeakEquations.cpp
    
    src/populist/ABConvertor.cpp
    src/populist/AbstractCirculantSolver.cpp
    src/populist/AbstractNonCirculantSolver.cpp
    src/populist/AbstractRateComputation.cpp
    src/populist/AbstractRebinner.cpp
    src/populist/AdaptiveHazard.cpp
    src/populist/CirculantSolver.cpp
    src/populist/InitializeAlgorithmGrid.cpp
    src/populist/IntegralRateComputation.cpp
    src/populist/InterpolationRebinner.cpp
    src/populist/LimitedNonCirculant.cpp
    src/populist/MatrixNonCirculant.cpp
    src/populist/MuSigmaScalarProduct.cpp
    src/populist/NonCirculantSolver.cpp
    src/populist/PolynomialCirculant.cpp
    src/populist/ProbabilityQueue.cpp
    src/populist/RefractiveCirculantSolver.cpp
    src/populist/VArray.cpp
    src/populist/ZeroLeakBuilder.cpp
    
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
    src/utilities/Log.cpp
    src/utilities/LogStream.cpp
    src/utilities/MPIProxy.cpp
    src/utilities/ParallelException.cpp
    src/utilities/ProgressBar.cpp
    src/utilities/TimeException.cpp
    src/utilities/Timer.cpp
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
# Cmake generated header files
#############################################################################
CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.cmake 
    ${CMAKE_CURRENT_BINARY_DIR}/config.hpp
)


#############################################################################
# testing
#############################################################################

IF (ENABLE_TESTING)
        ADD_SUBDIRECTORY(test)
ENDIF (ENABLE_TESTING)

install( FILES ${MPI_HEADERS}
    DESTINATION include/miind/MPILib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}mpi
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
