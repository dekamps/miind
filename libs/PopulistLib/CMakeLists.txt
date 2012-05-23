ADD_SUBDIRECTORY(Swig) # Given its own directory to be able to pass other compiler flag
project( ${MIIND_LIBRARY_PREFIX}populist )

if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module Populist)." )
  endif( NOT GSL_FOUND )
  link_directories( ${GSL_LIBRARY_RELEASE_DIRS} )
  set (LIB_TYPE STATIC )
else( WIN32 )
  link_directories( ${GSL_LIBRARY_DIRS} )
  include( RequireGSL )
  if ( APPLE )             #This is not necessary any more under leopard
    set( LIB_TYPE STATIC ) #but it makes no differnece
  else( APPLE )
    set( LIB_TYPE SHARED )
  endif( APPLE )
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )
include_directories( ${ROOT_INCLUDE_DIRS} )
link_directories( ${ROOT_LIBRARY_DIRS} )

set( populist_SRC
     ABConvertor.cpp
     AbstractCirculantSolver.cpp
     AbstractNonCirculantSolver.cpp
     AbstractRateComputation.cpp
     AbstractRebinner.cpp
     AbstractZeroLeakEquations.cpp
     AdaptationParameter.cpp
     AdaptiveHazard.cpp
     aeifdydt.cpp
     AEIFParameter.cpp
     AlgorithmBuilder.cpp
     CirculantSolver.cpp
     DelayActivityTest.cpp
     DoubleRebinner.cpp
     FitRateComputation.cpp
     InitializeAlgorithmGrid.cpp
     InitializePotentialVector.cpp
     IntegralRateComputation.cpp
     InterpolationRebinner.cpp
     LIFConvertor.cpp
     LIFZeroLeakEquations.cpp
     LimitedNonCirculant.cpp
     MatrixNonCirculant.cpp
     MuSigmaScalarProduct.cpp
     NonCirculantSolver.cpp
     NumericalZeroLeakEquations.cpp
     OldLifZeroLeakEquations.cpp
     OneDMParameter.cpp
     OneDMZeroLeakEquations.cpp
     OrnsteinUhlenbeckAlgorithm.cpp
     OrnsteinUhlenbeckParameter.cpp
     ParseResponseCurveMetaFile.cpp
     PlotParameter.cpp
     PolynomialCirculant.cpp
     PopulationAlgorithm.cpp
     PopulationGridControler.cpp
     PopulistSpecificParameter.cpp
     ProbabilityQueue.cpp
	 QIFParameter.cpp
	 RefractiveCirculantSolver.cpp
     Response.cpp
     SingleInputZeroLeakEquations.cpp
     SinglePeakRebinner.cpp
     SinglePopulationWorkFlow.cpp
     TestPopulist.cpp
     VArray.cpp
     VChebyshev.cpp
     WorkFlowAnalyzer.cpp
     WorkFlow.cpp
     WorkFlowTest.cpp
     ZeroLeakBuilder.cpp
 )

set( populist_HEADERS
     ABConvertor.h
     ABScalarProduct.h
     ABStruct.h
     AEIFParameter.h
     aeifdydt.h
     AbstractCirculantSolver.h
     AbstractNonCirculantSolver.h
     AbstractRateComputation.h
     AbstractRebinner.h
     AbstractZeroLeakEquations.h
     AdaptiveHazard.h
     AdaptationParameter.h
     BasicDefinitions.h
	 CharacteristicAlgorithm.h
	 CharacteristicAlgorithmCode.h
	 CharacteristicParameter.h
     CirculantSolver.h
     ConfigureException.h
     ConnectionSquaredProduct.h
     CreateTwoPopulationNetworkCode.h
     CreateTwoPopulationNetwork.h
     DelayActivityLib.h
     DelayActivityTest.h
     DoubleRebinner.h
     FitRateComputation.h
     InitialDensityParameter.h
     InitializeAlgorithmGrid.h
     InitializePotentialVector.h
     InputParameterSet.h
     IntegralRateComputation.h
     InterpolationRebinner.h
     LIFConvertor.h
     LIFZeroLeakEquations.h
     LimitedNonCirculant.h
     LocalDefinitions.h
     MatrixNonCirculant.h
     MuSigma.h
     MuSigmaScalarProduct.h
     NonCirculantSolver.h
     NoopRebinner.h
     NumericalZeroLeakEquations.h
	 NumericalZeroLeakParameter.h
     OldLifZeroLeakEquations.h
	 OneDMInputSetParameter.h
	 OneDMParameter.h
	 OneDMAlgorithm.h
	 OneDMAlgorithmCode.h
     OneDMZeroLeakEquations.h
     OrnsteinUhlenbeckAlgorithm.h
     OrnsteinUhlenbeckConnection.h
     OrnsteinUhlenbeckParameter.h
     OUDelayedConnection.h
     ParseResponseCurveMetaFile.h
     PlotParameter.h
     PolynomialCirculant.h
     PopulationAlgorithm.h
     PopulationAlgorithmCode.h
     PopulationGridController.h
	 PopulationGridControllerCode.h
     PopulationGrid.h
     PopulistException.h
     PopulistLib.h
     PopulistParameter.h
     PopulistSpecificParameter.h
     ProbabilityQueue.h
#	 ProbabilityQueueUnitTest.h
     QIFParameter.h
	 QIFAlgorithm.h
	 QIFAlgorithmCode.h
     RateException.h
	 RefractiveCirculantSolver.h
     Response.h
     ResponseParameterAmit.h
     ResponseParameterBrunel.h
     ResponseParameter.h
	 SingleInputZeroLeakEquations.h
     SinglePeakRebinner.h
     SinglePopulationWorkFlow.h
	 SpecialBins.h
     StampedProbability.h
     TestBinCalculationDefinitions.h
     TestDefinitions.h
     TestInhibitionDefinitions.h
     TestOmurtagDefinitions.h
     TestOUDefinitions.h
     TestPopulist.h
	 TestPopulistCode.h
     TestResponseCurveDefinitions.h
     TestZeroLeakDefinitions.h
     TestZeroLeakGaussDefinitions.h
     TransferTestDefinitions.h
     TwoPopulationTest.h
     VArray.h
     VChebyshev.h
     VMatrixCode.h
     VMatrix.h
     WorkFlow.h
     WorkFlowAnalyzer.h
     WorkFlowDefinitions.h
     WorkFlowTest.h
	 ZeroLeakBuilder.h
 )

add_library( ${MIIND_LIBRARY_PREFIX}populist ${LIB_TYPE} ${populist_SRC} ${populist_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}populist
    ${MIIND_LIBRARY_PREFIX}numtools
    ${MIIND_LIBRARY_PREFIX}dynamic
    ${MIIND_LIBRARY_PREFIX}net
    ${LIB_PREFIX}Core
    ${LIB_PREFIX}RIO
    ${LIB_PREFIX}Cint
    ${LIB_PREFIX}Hist
    ${LIB_PREFIX}Gpad
    ${LIB_PREFIX}Tree
    ${LIB_PREFIX}Table
    gsl
    ${CBLAS_PREFIX}cblas )


install( FILES ${populist_HEADERS}
    DESTINATION include/miind/PopulistLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}populist
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
