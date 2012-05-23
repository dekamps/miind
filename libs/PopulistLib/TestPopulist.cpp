// Copyright (c) 2005 - 2011 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software 
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY 
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifdef WIN32
#pragma warning(disable: 4800)
#endif

#include <gsl/gsl_math.h>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <TCanvas.h>
#include <TGraph.h>
#include <TH2F.h>
#include <TSVG.h>
#include <TFile.h>
#include <TFileIter.h>
#include <TROOT.h>
#include <TStyle.h>
#include "TestDefinitions.h"
#include "TestPopulist.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include "../DynamicLib/DynamicLib.h"
#include "../DynamicLib/DynamicLibTest.h"
#include "../SparseImplementationLib/SparseImplementationTest.h"
#include "AdaptiveHazard.h"
#include "AEIFParameter.h"
#include "aeifdydt.h"
#include "BasicDefinitions.h"
#include "CreateTwoPopulationNetworkCode.h"
#include "LIFConvertor.h"
#include "DelayActivityTest.h"
#include "DoubleRebinner.h"
#include "FitRateComputation.h"
#include "LimitedNonCirculant.h"
#include "MuSigmaScalarProduct.h"
#include "NonCirculantSolver.h"
#include "OneDMAlgorithmCode.h"
#include "OneDMParameter.h"
#include "PolynomialCirculant.h"
#include "PopulationAlgorithmCode.h"
#include "ProbabilityQueue.h"
#include "LocalDefinitions.h"
#include "MatrixNonCirculant.h"
#include "PopulationGridController.h"
#include "InitializeAlgorithmGrid.h"
#include "InterpolationRebinner.h"
#include "SingleInputZeroLeakEquations.h"
#include "TransferTestDefinitions.h"
#include "TestPopulistCode.h"
#include "TestOUDefinitions.h"
#include "TestZeroLeakDefinitions.h"
#include "TestZeroFluxLeakDefinitions.h"
#include "TestZeroLeakGaussDefinitions.h"
#include "TwoPopulationTest.h"
#include "VMatrixCode.h"
#include "VArray.h"
#include "VChebyshev.h"
#include "ZeroLeakBuilder.h"

using namespace std;
using namespace UtilLib;
using namespace PopulistLib;
using namespace DynamicLib;
using namespace SparseImplementationLib;

using NumtoolsLib::IsApproximatelyEqualTo;
using NumtoolsLib::Precision;

TestPopulist::TestPopulist (boost::shared_ptr<ostream> p):
LogStream (p)
{
}

TestPopulist::~TestPopulist ()
{
}

bool TestPopulist::Execute ()
{
	ofstream test("test/bla");
	if (! test)
	{
		cout << "Please create a directory called 'test' here" << endl;
		return true;
	}

	if (! InnerProductTest() )
		return false;
	Record("InnerProductTest succeeded");

	if (! BinCalculationTest() )
		return false;
	Record("BinCalculationTest succeeded");

	if (! ZeroLeakTest() )
		return false;
	Record("ZeroLeakTest succeeded");

	// ZeroLeakFluxTest was here, but has not been properly implemented

	if (! InitialDensityTest() )
		return false;
	Record("InitialDensityTest succeeded");

	if ( ! GammaZTest() )
		return false;
	Record("GammaZTest succeeded");

	if ( ! Vkj3Test() )
		return false;
	Record("VkjTest succeeded");

	if ( ! NonCirculantTransferTest() )
		return false;
	Record("NonCirculantTransferTest succeeded");

	if ( ! GenerateVLookUp() )
		return false;
	Record("GenerateVLookUp succeeded");

	if (! VArrayTest() )
		return false;
	Record("VArrayTest succeeded");

	if ( ! ChebyshevVTest() )
		return false;
	Record ("ChebyshevVTest succeeded");

	if (! DoubleRebinnerTest() )
		return false;
	Record("DoubleRebinnerTest succeeded");

	if (! PotentialToBinTest() )
		return false;
	Record("PotentialToBinTest succeeded");

	if (! GenerateVDataTest() )
		return false;
	Record("GenerateVDataTest succeeded");

	if (! InputConvertorTest () )
		return false;
	Record("InputConvertorTest succeeded");

	if (! this->OldOmurtagetAlTest() )
		return false;
	Record("OldOmurtagetAlTest succeeded");

	if (! OmurtagetAlTest ())
		return false;
	Record("OmurtagetAlTest succeeded");

	if (! this->SingleInputZeroLeakEquationsTest() )
		return false;
	Record("SingleInputZeroLeakEquationsTest succeeded");

	if (! OmurtagRefractiveZeroTest())
		return false;
	Record("OmurtagRefractiveZeroTest succeeded");

	if (! OmurtagRefractiveTest() )
		return false;
	Record("OmurtagRefractiveTest succeded");

	if (! OmurtagNumericalTest ())
		return false;
	Record("OmurtagetNumericalTest succeeded");

	if (! OmurtagNumericalRefractiveTest() )
		return false;
	Record(" NumericalOmurtagRefractiveTest");

	// This is decomissioned at the moment, because the usefulness of the fitting procedure must be reassessed
//	if (! OmurtagFitTest () )
//		return false;
//	Record ("OmurtagFitTest succeeded");

	if (! OmurtagDoubleTest () )
		return false;
	Record("OmurtagDoubleTest");

	if ( ! OmurtagPolynomialTest() )
		return false;
	Record("OmurtagPolynomialTest succeeded");

	if ( ! OmurtagMatrixTest() )
		return false;
	Record("OmurtagMatrixTest succeeded");

	if (! ZeroLeakBuilderTest() )
		return false;
	Record("ZeroLeakBuilderTest succeeded");

	if ( ! TestResponseCurveSingle(false) )
		return false;
	Record(" TestResponseCurveNonRefractiveSingle succeeded");

	if ( ! TestResponseCurveDouble(false) )
		return false;
	Record("TestResponseCurveNonRefractiveDouble succeeded");

	if ( ! TestResponseCurveSingle(true) )
		return false;
	Record(" TestResponseCurveSingle succeeded");

	if ( ! TestResponseCurveDouble(true) )
		return false;
	Record("TestResponseCurveDouble succeeded");

	if (! InhibitionTest() )
		return false;
	Record("Inhibitiontest succeeded");

	if (! this->TwoPopulationTest() )
		return false;
	Record("TwoPopulationTest succeeded");

	if ( ! PrintResponseCurve() )
		return false;
	Record("PrintResponseCurve succeeded");

	if ( ! ScalarProductTest() )
		return false;
	Record("ScalarProductTest succeeded");

	if ( ! AEIFIntegratorTest() )
		return false;
	Record("AEIFIntegratorTest succeeded");

	if (!  HazardFunctionTest() )
		return false;
	Record("HazardFunctionTest succeeded");

	if ( ! OneDMTest() )
		return false;
	Record("OneDMTest succeeded");

	if ( ! BalancedExample() )
		return false;
	Record("BalancedExample succeeded");

	if (! this->StreamPopulationAlgorithmIn() )
		return false;
	Record("StreamPopulationAlgorithmIn");

	if (! this->StreamPopulationAlgorithmOut() )
		return false;
	Record("StreamPopulationAlgorithmOut");

	if (! this->BuildPopulationAlgorithm() )
		return false;
	Record("BuildPopulationAlgorithm succeeded");

	if (! StreamOUAlgorithmOut() )
		return false;
	Record("StreamOUAlgorithmOut succeeded");

	if (!  StreamOUAlgorithmIn() )
		return false;
	Record("StreamOUAlgorithmIn succeeded");

	if (! BuildOUAlgorithm() )
		return false;
	Record("BuildOUAlgorithm succeeded");

	if (! RootFileInterpreterTest() )
		return false;
	Record("RootFileInterpreterTest");

	ProcessResults();

	cout << "Ended plots" << endl;
	return true;
}

bool TestPopulist::InnerProductTest () const 
{
	// deprecated, PopulationAlgorithms defer the calculation
	// of inner products now to ScalarProduct objects, see the relevant
	// tests: ScalarProductTest
	/* 
	Pop_DynamicNode node1;
	Pop_DynamicNode node2;
	Pop_DynamicNode node3;

	node1.SetValue (1.0);
	node2.SetValue (1.0);
	node3.SetValue (1.0);

	typedef pair < Pop_DynamicNode *, PopulationConnection > connection;
	typedef Pop_DynamicNode::predecessor_iterator predecessor_iterator;

	vector < connection > vector_of_connections;

	PopulationConnection connection_1 (1, 1);
	PopulationConnection connection_2 (1, 2);
	PopulationConnection connection_3 (1, 3);
		
	vector_of_connections.push_back
	(
		connection
		(
			&node1, 
			connection_1
		)
	);

	vector_of_connections.push_back
	(
		connection
		(
			&node2, 
			connection_2
		)
	);

	vector_of_connections.push_back
	(
		connection
		(
			&node3, 
			connection_3
		)
	);

	// Need a concrete Algorithm to calculate the inner product
	PopulationParameter par_pop;
	PopulistSpecificParameter par_spec;

	PopulationAlgorithm<> algorithm(PopulistParameter(par_pop,par_spec));

	// clumsy notation, because this Algorithm isn't in a Node

	connection* p = &(*vector_of_connections.begin());
//	double f_inner_product = 
//		algorithm.InnerProduct
//		(
//			predecessor_iterator(&(*vector_of_connections.begin ())),
//			predecessor_iterator (&(*vector_of_connections.end ()))
//		);

//	     if (! Numtools::IsApproximatelyEqualTo
//			(
//				f_inner_product, 
//				6.0, 
//				1e-10
//			)
//		)
//		       return false;
//
*/
		return true;
     }

bool TestPopulist::BinCalculationTest () const 
{
	//! This routine runs the generic one population test without any effective input.
	//! Purpose is to monitor the expansion of the grid for a leaky-integrate-and-fire 
	//! population. As the evolution progresses,
	//! 1) the initial density, which is put at the reversal potention should stay there.
	//! 2) the highest bin should always correspond to the threshold potential
	//! 3) the number of bins should increase in line with exponential expansion of the grid
	//! This test only produces the root files with the simulation results. Root Macro Validation.cpp
	//! should be loaded and Validation() should be called to validate the results of BinCaluclationTest

	return 
		TestPopulist::GenericOnePopulationTest
		(
			BINCALCULATION_RATE,
			BINCALCULATION_EFFICACY,
			BINCALCULATION_PARAMETER,
			BINCALCULATION_PARAMETER_RUN,
			BINCALCULATION_SPECIFIC
		);
}

     bool TestPopulist::ZeroLeakTest () const
     {
	     return TestPopulist::GenericOnePopulationTest
			(
				ZEROLEAK_RATE,
				ZEROLEAK_EFFICACY,
				ZEROLEAK_PARAMETER,
				ZEROLEAK_PARAMETER_RUN,
				ZEROLEAK_SPECIFIC
			);
	     return true;
     }

	 bool TestPopulist::ZeroLeakFluxTest() const
	 {
		// The naive MatrixNonCirculantSolver and CirculantSolver should give identical results

		valarray<double> val_stand(N_ZEROLEAKFLUX_BINS);
		valarray<double> val_mat(N_ZEROLEAKFLUX_BINS);
		val_stand    = 0.0;
		val_mat      = 0.0;
		for (Index icomp = 0; icomp < ZEROLEAKFLUX_H; icomp++)
		{
			val_stand[icomp] = icomp;
			val_mat[icomp]   = icomp;
		}
		NonCirculantSolver non_standard;
		MatrixNonCirculant non_matrix;

		// First for positive H
		InputParameterSet par_input;

		par_input._H_exc		 = ZEROLEAKFLUX_H;
		par_input._n_noncirc_exc = N_ZEROLEAKFLUX_BINS;
	
		non_standard.Configure
		(
			val_stand,
			par_input
		);

		non_matrix.Configure
		(
			val_mat,
			par_input
		);

		non_standard.ExecuteExcitatory
		(
			N_ZEROLEAKFLUX_BINS,
			ZEROLEAKFLUX_TAU
		);
	
		non_matrix.ExecuteExcitatory
		(
			N_ZEROLEAKFLUX_BINS,
			ZEROLEAKFLUX_TAU
		);

		non_matrix.ExecuteInhibitory
		(
			N_ZEROLEAKFLUX_BINS,
			ZEROLEAKFLUX_TAU
		);
		for (int i = 0; i < static_cast<int>(N_ZEROLEAKFLUX_BINS); i++)
			if (val_mat[i] != val_stand[i])
				return false;
	
		Timer t;

		int n_steps = 1000000;
		for (int t_st = 0; t_st < n_steps; t_st++)
		{
			
			non_standard.ExecuteExcitatory
			(
				N_ZEROLEAKFLUX_BINS,
				ZEROLEAKFLUX_TAU
			);
		}


		for (int t_ma = 0; t_ma < n_steps; t_ma++)
		{
			non_matrix.ExecuteExcitatory
			(
				N_ZEROLEAKFLUX_BINS,
				ZEROLEAKFLUX_TAU
			);

			non_matrix.ExecuteInhibitory
			(
				N_ZEROLEAKFLUX_BINS,
				ZEROLEAKFLUX_TAU
			);			
		}
	

		return true;
	 }

	 bool TestPopulist::OldOmurtagetAlTest() const
	 {
		const RootReportHandler 
			OLDOMURTAG_HANDLER
			(
				"test/old_omurtag.root",
				TEST_RESULTS_ON_SCREEN,
				true,
				CANVAS_OMURTAG
			);

		const SimulationRunParameter 
			OLDOMURTAG_PARAMETER_RUN
			(
				OLDOMURTAG_HANDLER,
				OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
				OMURTAG_T_BEGIN,
				OMURTAG_T_END,
				OMURTAG_T_REPORT,
				OMURTAG_T_REPORT,
				OMURTAG_T_NETWORK_STEP,
				"test/omurtag_old.log"
			);
		PopulistSpecificParameter 
			par_spec
			(
				OMURTAG_V_MIN,
				OMURTAG_NUMBER_INITIAL_BINS,
				OMURTAG_NUMBER_OF_BINS_TO_ADD,
				OMURTAG_INITIAL_DENSITY,
				OMURTAG_EXPANSION_FACTOR,
				"OldLIFZeroLeakEquations"
			);

		TestPopulist::GenericOnePopulationTest
		(    
			OMURTAG_RATE,
			OMURTAG_EFFICACY,
			OMURTAG_PARAMETER,
			OLDOMURTAG_PARAMETER_RUN,
			par_spec,
			true
		);

		 return true;
	 }

     bool TestPopulist::OmurtagetAlTest () const 
     {
		 // This test runs the default algorithm with the standard Circulant and NonCirculant solvers. It was used to produce
		 // the figures in 'The State of MIIND', so needs to be retained out of historic acccountability. 
		const RootReportHandler 
			SingleOmurtagHandler
			(
				"test/single_omurtag.root",
				TEST_RESULTS_ON_SCREEN,
				true,
				CANVAS_OMURTAG
			);

		const SimulationRunParameter 
			OMURTAG_SINGLE_PARAMETER_RUN
			(
				SingleOmurtagHandler,
				OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
				OMURTAG_T_BEGIN,
				OMURTAG_T_END,
				OMURTAG_T_REPORT,
				OMURTAG_T_REPORT,
				OMURTAG_T_NETWORK_STEP,
				"test/omurtag_single.log"
			);
		PopulistSpecificParameter 
			par_spec
			(
				OMURTAG_V_MIN,
				OMURTAG_NUMBER_INITIAL_BINS,
				OMURTAG_NUMBER_OF_BINS_TO_ADD,
				OMURTAG_INITIAL_DENSITY,
				OMURTAG_EXPANSION_FACTOR,
				"SingleInputZeroLeakEquations"
			);
	     return TestPopulist::GenericOnePopulationTest
			(
		     
				OMURTAG_RATE,
				OMURTAG_EFFICACY,
				OMURTAG_PARAMETER,
				OMURTAG_SINGLE_PARAMETER_RUN,
				par_spec,
				true
			);
     }

	 bool TestPopulist::SingleInputZeroLeakEquationsTest() const
	 {
		 // Same test as the OmurtagetAlTest but with SingleInputZeroLeakEquations, rather than LIFZeroLeakEquations. This is
		 // a theoretically optimally efficient version of the algorithm.

		 PopulistSpecificParameter 
			 par_spec
			 (
				OMURTAG_V_MIN,
				OMURTAG_NUMBER_INITIAL_BINS,
				OMURTAG_NUMBER_OF_BINS_TO_ADD,
				OMURTAG_INITIAL_DENSITY,
				OMURTAG_EXPANSION_FACTOR,
				"SingleInputZeroLeakEquations"
			);

	     return TestPopulist::GenericOnePopulationTest
			(
		     
				OMURTAG_RATE,
				OMURTAG_EFFICACY,
				OMURTAG_PARAMETER,
				OMURTAG_PARAMETER_RUN,
				par_spec,
				true
			);
	 }

	 bool TestPopulist::OmurtagRefractiveZeroTest() const 
	 {
		 // This test runs the default algorithm with the standard Circulant and NonCirculant solvers. It was used to produce
		 // the figures in 'The State of MIIND', so needs to be retained out of historic acccountability. 
		const RootReportHandler 
			SingleOmurtagHandler
			(
				"test/omurtag_refractive_zero.root",
				TEST_RESULTS_ON_SCREEN,
				true,
				CANVAS_OMURTAG
			);

		const SimulationRunParameter 
			OMURTAG_SINGLE_PARAMETER_RUN
			(
				SingleOmurtagHandler,
				OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
				OMURTAG_T_BEGIN,
				OMURTAG_T_END,
				OMURTAG_T_REPORT,
				OMURTAG_T_REPORT,
				OMURTAG_T_NETWORK_STEP,
				"test/omurtag_refractive_zero.log"
			);
		PopulistSpecificParameter 
			par_spec
			(
				OMURTAG_V_MIN,
				OMURTAG_NUMBER_INITIAL_BINS,
				OMURTAG_NUMBER_OF_BINS_TO_ADD,
				OMURTAG_INITIAL_DENSITY,
				OMURTAG_EXPANSION_FACTOR,
				"SingleInputZeroLeakEquations",
				"RefractiveCirculantSolver",
				"NonCirculantSolver"
			);

	     return TestPopulist::GenericOnePopulationTest
			(
		     
				OMURTAG_RATE,
				OMURTAG_EFFICACY,
				OMURTAG_PARAMETER,
				OMURTAG_SINGLE_PARAMETER_RUN,
				par_spec,
				true
			);
	 }

	bool TestPopulist::OmurtagRefractiveTest() const 
	 {
		 // This test runs the default algorithm with the standard Circulant and NonCirculant solvers. It was used to produce
		 // the figures in 'The State of MIIND', so needs to be retained out of historic acccountability. 
		const RootReportHandler 
			SingleOmurtagHandler
			(
				"test/omurtag_refractive.root",
				TEST_RESULTS_ON_SCREEN,
				true,
				CANVAS_OMURTAG
			);

		const SimulationRunParameter 
			OMURTAG_SINGLE_PARAMETER_RUN
			(
				SingleOmurtagHandler,
				OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
				OMURTAG_T_BEGIN,
				OMURTAG_T_END,
				OMURTAG_T_REPORT,
				OMURTAG_T_REPORT,
				OMURTAG_T_NETWORK_STEP,
				"test/omurtag_refractive.log"
			);
		PopulistSpecificParameter 
			par_spec
			(
				OMURTAG_V_MIN,
				OMURTAG_NUMBER_INITIAL_BINS,
				OMURTAG_NUMBER_OF_BINS_TO_ADD,
				OMURTAG_INITIAL_DENSITY,
				OMURTAG_EXPANSION_FACTOR,
				"SingleInputZeroLeakEquations",
				"RefractiveCirculantSolver",
				"NonCirculantSolver"
			);

		PopulationParameter par = OMURTAG_PARAMETER;
		par._tau_refractive = 5e-3;

	    return TestPopulist::GenericOnePopulationTest
			(
		     
				OMURTAG_RATE,
				OMURTAG_EFFICACY,
				par,
				OMURTAG_SINGLE_PARAMETER_RUN,
				par_spec,
				true
			);
	 }

bool TestPopulist::OmurtagNumericalTest() const
{

	const RootReportHandler 
		SingleNumericalOmurtagHandler
		(
			"test/omurtag_numerical.root",
			TEST_RESULTS_ON_SCREEN,
			true,
			CANVAS_OMURTAG
		);

		const SimulationRunParameter 
			OmurtagSingleRunNumerical
			(
				SingleNumericalOmurtagHandler,
				OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
				OMURTAG_T_BEGIN,
				OMURTAG_T_END,
				OMURTAG_T_REPORT,
				OMURTAG_T_REPORT,
				OMURTAG_T_NETWORK_STEP,
				"test/omurtag_numerical.log"
			);

		PopulistSpecificParameter 
			par_spec
			(
				OMURTAG_V_MIN,
				OMURTAG_NUMBER_INITIAL_BINS,
				OMURTAG_NUMBER_OF_BINS_TO_ADD,
				OMURTAG_INITIAL_DENSITY,
				OMURTAG_EXPANSION_FACTOR,
				"NumericalZeroLeakEquations"
			);
	GenericOnePopulationTest
	(
		OMURTAG_RATE,
		OMURTAG_EFFICACY,
		OMURTAG_PARAMETER,
		OmurtagSingleRunNumerical,
		par_spec,
		true
	);

	return true;
}

	bool TestPopulist::OmurtagNumericalRefractiveTest() const 
	{
		const RootReportHandler 
			SingleNumericalRefractiveOmurtagHandler
			(
				"test/omurtag_numerical_refractive.root",
				TEST_RESULTS_ON_SCREEN,
				true,
				CANVAS_OMURTAG
			);


		const SimulationRunParameter 
			OmurtagSingleRunNumericalRefractive
			(
				SingleNumericalRefractiveOmurtagHandler,
				OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
				OMURTAG_T_BEGIN,
				OMURTAG_T_END,
				OMURTAG_T_REPORT,
				OMURTAG_T_REPORT,
				OMURTAG_T_NETWORK_STEP,
				"test/omurtag_numerical_refractive.log"
			);

		PopulistSpecificParameter 
			par_spec
			(
				OMURTAG_V_MIN,
				OMURTAG_NUMBER_INITIAL_BINS,
				OMURTAG_NUMBER_OF_BINS_TO_ADD,
				OMURTAG_INITIAL_DENSITY,
				OMURTAG_EXPANSION_FACTOR,
				"NumericalZeroLeakEquations"
			);

		PopulationParameter par = OMURTAG_PARAMETER;
		par._tau_refractive = 5e-3;

	    return TestPopulist::GenericOnePopulationTest
			(		     
				OMURTAG_RATE,
				OMURTAG_EFFICACY,
				par,
				OmurtagSingleRunNumericalRefractive,
				par_spec,
				true
			);
		return true;
	}

	bool TestPopulist::OmurtagFitTest() const
	{

	// Standard Omega test, with fit rate calculation

	const SimulationRunParameter 
		OMURTAG_PARAMETER_FIT
		(
			OMURTAG_FIT_HANDLER,
			OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
			OMURTAG_T_BEGIN,
			OMURTAG_T_END,
			OMURTAG_T_REPORT,
			OMURTAG_T_REPORT,
			OMURTAG_T_NETWORK_STEP,
			"test/omurtagfit.log"
		);

		GenericOnePopulationTest
		(
			OMURTAG_RATE,
			OMURTAG_EFFICACY,
			OMURTAG_PARAMETER,
			OMURTAG_PARAMETER_FIT,
			OMURTAG_SPECIFIC,
			false
		);
	
	     return true;
     }


bool TestPopulist::GammaZTest () const 
{
	VMatrix <FULL_CALCULATION> solver;

	int maximum_number_gamma_z_values = 120;
	int n_circulant = 50;

	solver.FillGammaZ
	(
		maximum_number_gamma_z_values,
		n_circulant,
		5.0
	);

	int l = 0;
	complex <double> omega_kt = 
		       exp (complex <double >
			(
				l * 2.0 * M_PI / static_cast<double>(n_circulant),
				0
			) *complex < double >
				(
					0.0,
					1.0)
				)* 5.0;

	       complex < double >test = solver.Gamma (1, 0);
	     if (
			!IsApproximatelyEqualTo
			(
				test.real (),
				(complex <double>(1.0, 0.0) - exp (-omega_kt)).real (),
				1e-10)
		)
		       return false;


	       test = solver.Gamma (1, 14);
	       l = 14;
	       omega_kt =
		     exp (complex <double >
				(
					l * 2.0 * M_PI / static_cast<double>(n_circulant),0)*complex < double >(0.0, 1.0)
				)*5.0;
	     if (
			!IsApproximatelyEqualTo
			(
				test.real (),
				(complex < double >(1.0, 0.0) - exp (-omega_kt)).real (),
				1e-10
			)
		)
		return false;
	
	return true;
     }


     bool TestPopulist::Vkj3Test () const 
     {
	     // A simple test that allows quick checking in the debugger
	     // Compare to the output of MATLAB routines

	     VMatrix <FULL_CALCULATION> solver;

	     Number number_circulant = 3;
	     Time tau = 0.1;

	     if (
			!IsApproximatelyEqualTo
			(
				solver.V 
				(
					number_circulant, 
					0, 
					0, 
					tau
				), 
				0.09048751197746,
				1e-11
			)
		 ||	!IsApproximatelyEqualTo 
			(
				solver.V
				(
					number_circulant, 
					0, 
					1, 
					tau
				),
				0.00452426249352,
				1e-11
			)
		 ||	!IsApproximatelyEqualTo
			(
				solver.V
				(
					number_circulant, 
					0, 
					2, 
					tau
				),
				0.00015080749306,
				1e-11
			)
		 ||	!IsApproximatelyEqualTo
			(
				solver.V
				(
					number_circulant, 
					1, 
					0, 
					tau
				),
				0.00452426249352, 
				1e-11
			)
		 ||	!IsApproximatelyEqualTo
			(
				solver.V
				(
					number_circulant, 
					1, 
					1, 
					tau
				),
				0.00015080749306, 
				1e-11
			)
		 ||	!IsApproximatelyEqualTo
			(
				solver.V
				(
					number_circulant,
					1, 
					2, 
					tau
				),
				0.00000377017386, 
				1e-11
			)
		 ||	!IsApproximatelyEqualTo
			(
				solver.V
				(
					number_circulant, 
					2, 
					0, 
					tau
				),
				0.00015080749306,
				1e-11
			)
		 ||	!IsApproximatelyEqualTo
			(
				solver.V
				(
					number_circulant, 
					2, 
					1, 
					tau
				),
				0.00000377017386, 
				1e-11
			)
		 ||	!IsApproximatelyEqualTo
			(
				solver.V
				(
					number_circulant, 
					2, 
					2, 
					tau
				),
				0.00000007540344, 
				1e-11
			)
		)
		       return false;

	     // also test an even solution, with number of non-circulant > number of circulant

	       number_circulant = 4;
	       tau = 1;


	     if (!IsApproximatelyEqualTo
		 (solver.V (number_circulant, 0, 0, tau), 0.37094611701740,
		  1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 0, 1, tau),
					     0.18445076563595, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 0, 2, tau),
					     0.06138624136429, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 0, 3, tau),
					     0.01533743481092, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 1, 0, tau),
					     0.18445076563595, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 1, 1, tau),
					     0.06138624136429, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 1, 2, tau),
					     0.01533743481092, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 1, 3, tau),
					     0.00306667584596, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 2, 0, tau),
					     0.06138624136429, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 2, 1, tau),
					     0.01533743481092, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 2, 2, tau),
					     0.00306667584596, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 2, 3, tau),
					     0.00051104505023, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 3, 0, tau),
					     0.01533743481092, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 3, 1, tau),
					     0.00306667584596, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 3, 2, tau),
					     0.00051104505023, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 3, 3, tau),
					     0.00007300116905, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 0, 4, tau),
					     0.00306667584596, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 1, 4, tau),
					     0.00051104505023, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 2, 4, tau),
					     0.00007300116905, 1e-11)
		 || !IsApproximatelyEqualTo (solver.
					     V (number_circulant, 3, 4, tau),
					     0.00000912476211, 1e-11)
		     )
		       return false;


	       return true;
     }

     bool TestPopulist::NonCirculantTransferTest () const 
     {
	     return TestPopulist::GenericOnePopulationTest
			(
				TRANSFER_RATE,
				TRANSFER_EFFICACY,
				TRANSFER_PARAMETER,
				TRANSFER_PARAMETER_RUN,
				TRANSFER_SPECIFIC
			);
	     return true;
     }

     bool TestPopulist::InitialDensityTest () const
     {
	     PopulationParameter 
		     parameter_population 
		     (
				1,	// theta
				0,	// V_reset
				0,	// V_reversal
				0,	// no refractive period
				50e-3	// 50 ms or 20 s^-1
		     );

	     int number_of_bins = 5;
	     Potential v_min = 0;

	     InitialDensityParameter parameter_density (0, 0);

	     InitializeAlgorithmGrid init;
	     AlgorithmGrid grid = 
			init.InitializeGrid 
			(
				number_of_bins,
				v_min,
				parameter_population,
				parameter_density
			);

	       vector<double> vector_density = grid.ToStateVector ();

	     if (	vector_density[0] != number_of_bins	||
			vector_density[1] != 0			||
			vector_density[2] != 0			||
			vector_density[3] != 0			|| 
			vector_density[4] != 0
		)
		       return false;

	       parameter_density._mu = 0.25;

	       InitializeAlgorithmGrid init2;
	       grid = init2.InitializeGrid
			(
				number_of_bins,
				v_min, 
				parameter_population, 
				parameter_density
			);

	       vector_density = grid.ToStateVector ();

	     if (vector_density[0] != 0 ||
		 vector_density[1] != number_of_bins ||
		 vector_density[2] != 0 ||
		 vector_density[3] != 0 || vector_density[4] != 0)
		       return false;

	       parameter_density._mu = 0.5;

	       InitializeAlgorithmGrid init3;
	       grid = init3.InitializeGrid
			(
				number_of_bins,
				v_min, 
				parameter_population, 
				parameter_density
			);

	       vector_density = grid.ToStateVector ();

	     if (vector_density[0] != 0 ||
		 vector_density[1] != 0 ||
		 vector_density[2] != number_of_bins ||
		 vector_density[3] != 0 || vector_density[4] != 0)
		       return false;

	       parameter_density._mu = 0.75;

	      
	       grid = init.InitializeGrid
			(
				number_of_bins,
				v_min, 
				parameter_population, 
				parameter_density
			);

	       vector_density = grid.ToStateVector ();

	     if (vector_density[0] != 0 ||
		 vector_density[1] != 0 ||
		 vector_density[2] != 0 ||
		 vector_density[3] != number_of_bins
		 || vector_density[4] != 0)
		       return false;

	       parameter_density._mu = 1.0;

	       grid = init.InitializeGrid
			(
				number_of_bins,
				v_min, 
				parameter_population, 
				parameter_density
			);

	       vector_density = grid.ToStateVector ();

	     if (vector_density[0] != 0 ||
		 vector_density[1] != 0 ||
		 vector_density[2] != 0 ||
		 vector_density[3] != 0
		 || vector_density[4] != number_of_bins)
		       return false;


	       parameter_density._mu = 0.3;

	       grid = init.InitializeGrid
			(
				number_of_bins,
				v_min, 
				parameter_population, 
				parameter_density
			);

	       vector_density = grid.ToStateVector ();

	     if (vector_density[0] != 0 ||
		 vector_density[1] != number_of_bins ||
		 vector_density[2] != 0 ||
		 vector_density[3] != 0 || vector_density[4] != 0)
		       return false;


	       return true;
     }

 
     bool TestPopulist::GenerateVLookUp () const
     {
	     VMatrix < FULL_CALCULATION > vmatrix_dumb;

//	     cout << vmatrix_dumb.V (5, 2, 2, 0.0425) << endl;

	     return true;
     }

	bool TestPopulist::VArrayTest () const
	{
		VMatrix < FULL_CALCULATION > vmatrix_dumb;

		Time tau = 0.5;
		Number number_circulant_bins = 30;
		Number number_non_circulant_bins = 70;
		for 
		(
			int index_circulant = 0;
			index_circulant < static_cast <int >(number_circulant_bins); 
			index_circulant++
		)
			for
			(
				int index_non_circulant = 0;
				index_non_circulant < static_cast <int >(number_non_circulant_bins);
				index_non_circulant++
			)
			{
				double v_dumb =
					vmatrix_dumb.V
					(
						number_circulant_bins,
						index_circulant,
						index_non_circulant,
						tau
					);

				VArray v_array;
				v_array.FillArray
				(
					number_circulant_bins,
					number_non_circulant_bins,
					tau
				);

				double v_smart = 
				v_array.V 
				(
					index_circulant,
					index_non_circulant
				);

				if (fabs (v_dumb - v_smart) > 1e-12)
				       return false;
			}

	     return true;
     }


bool TestPopulist::InhibitionTest() const
{
	return TestPopulist::GenericOnePopulationTest
		(   
			INHIBITION_RATE,
			INHIBITION_EFFICACY,
			INHIBITION_PARAMETER,
			INHIBITION_PARAMETER_RUN,
			INHIBITION_SPECIFIC
		);
}

bool TestPopulist::ZeroLeakGaussTest() const
{
	return TestPopulist::GenericTwoPopulationTest
		(   
			ZEROLEAKGAUSS_RATE,
			ZEROLEAKGAUSS_EFFICACY,
			ZEROLEAKGAUSS_RATE,
			-ZEROLEAKGAUSS_EFFICACY,
			ZEROLEAKGAUSS_PARAMETER,
			ZEROLEAKGAUSS_PARAMETER_RUN,
			ZEROLEAKGAUSS_SPECIFIC,
			true
		);
}

bool TestPopulist::DoubleRebinnerTest() const
{
	return true;
}

bool TestPopulist::OmurtagDoubleTest() const
{
	// Classical Omurtag test, but with double rebinner and expansion factor of 2

	InitializeAlgorithmGrid init;
//TODO: check out why this variable is not used
//	double expansion_factor = 
		init.ExpansionFactorDoubleRebinner
		(
			OMURTAG_NUMBER_INITIAL_BINS,
			OMURTAG_V_MIN,
			OMURTAG_PARAMETER
		);

	const SimulationRunParameter 
		run_parameter
		(
			OMURTAG_DOUBLE_REBIN_HANDLER,
			OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
			OMURTAG_T_BEGIN,
			OMURTAG_T_END,
			OMURTAG_T_REPORT,
			OMURTAG_T_REPORT,
			OMURTAG_T_NETWORK_STEP,
			"test/omurtagdouble.log"
		);

	return TestPopulist::GenericOnePopulationTest
		(  
			OMURTAG_RATE,
			OMURTAG_EFFICACY,
			OMURTAG_PARAMETER,
			run_parameter,
			OMURTAG_SPECIFIC,
			true
		);
}

bool TestPopulist::PotentialToBinTest() const
{

	PopulationAlgorithm_<PopulationConnection> pop(PopulistParameter(OMURTAG_PARAMETER,OMURTAG_SPECIFIC));

	pop.Configure(OMURTAG_PARAMETER_RUN);

	for 
	(
		Index i = 0;
		i < OMURTAG_NUMBER_INITIAL_BINS;
		i++
	)
	{
		Potential v = pop.BinToCurrentPotential(i);
		Index j = pop.CurrentPotentialToBin(v);
		if ( i != j )
			return false;
	}

	return true;
}

bool TestPopulist::OrnsteinUhlenbeckProcessTest() const
{
		return TestPopulist::GenericTwoPopulationTest
		(   
			OUTEST_RATE,
			OUTEST_EFFICACY,
			OUTEST_RATE,
			-OUTEST_EFFICACY,
			OUTEST_PARAMETER,
			OUTEST_PARAMETER_RUN,
			OUTEST_SPECIFIC,
			true
		);
}

bool TestPopulist::TwoPopulationTest() const
{
	NodeId id_cortical_background;
	NodeId id_excitatory_main;
	NodeId id_inhibitory_main;
	NodeId id_rate;
	OU_Network network = 
		CreateTwoPopulationNetwork<PopulationAlgorithm_<PopulationConnection>, DynamicNetworkImplementation<PopulationAlgorithm_<PopulationConnection>::WeightType> >
		(
			&id_cortical_background,
			&id_excitatory_main,
			&id_inhibitory_main,
			&id_rate,
			TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER_POP,
			TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER_POP
		);


	TWOPOP_HANDLER.AddNodeToCanvas(id_excitatory_main);
	TWOPOP_HANDLER.AddNodeToCanvas(id_inhibitory_main);

	bool b_configure = 
		network.ConfigureSimulation
			(
				TWOPOP_PARAMETER
			);

	if (! b_configure )
		return false;

	bool b_evolve = network.Evolve();

	return b_evolve;
}

bool TestPopulist::ChebyshevVTest() const
{
	VChebyshev v_cheb;

//	cout << v_cheb.V(0,0,0.3) << endl;
//	cout << v_cheb.V(0,1,0.3) << endl;
//	cout << v_cheb.V(1,0,0.3) << endl;

	VArray v_array;
	v_array.FillArray(N_CIRC_MAX_CHEB,N_NON_CIRC_MAX_CHEB, 0.3);

//	cout << v_array.V(0,0) << endl;
//	cout << v_array.V(1,0) << endl;
//	cout << v_array.V(0,1) << endl;


	return true;
}

bool TestPopulist::GenerateVDataTest() const
{
	// Purpose: Generate the elements V_kj for various n, k-j, tau
	// Used for the generation of the the plots of these elements.
	// Adapt the constants to produce other step sizes or limits. Requires
	// recompilation!

	// Author: Marc de Kamps
	ofstream stream((STR_TEST_DIRECTORY + string("VDATA")).c_str());

	stream.precision(10);

	Time log_tmin = log10(VDATA_TMIN);
	Time log_tmax = log10(VDATA_TMAX);

	for (int n_circ = 2; n_circ < VDATA_NCIRC; n_circ++)
		for ( int j = 0; j < VDATA_JMAX; j++)
			for ( int t = 0; t < VDATA_NSTEPS; t++)
			{
				Time tau = pow(10.0,((log_tmax - log_tmin)/VDATA_NSTEPS)*t + log_tmin);
				VArray v_array;

				v_array.FillArray(n_circ ,VDATA_NCIRC + 10,tau);
		
				stream << tau << " " << n_circ << " " << j << " " << v_array.V(0,j) << "\n";
			}
	return true;
}

bool TestPopulist::OmurtagPolynomialTest() const
{
	// This test is the same as  OmurtagetalTest, but the Circulant is replaced with a PolynomialCirculant, and the NonCirculant
	// is replaced with the LimitedNonCirculant. It still runs with LIFZeroLeakEquations, so in that sense is still not optimal.
	// It runs for a simulation time of 100 s, to estimate the real time it takes. The outputs must be evaluated with
	GenericOnePopulationTest
	(
		OMURTAG_RATE,
		OMURTAG_EFFICACY,
		OMURTAG_PARAMETER,
		OMURTAG_PARAMETER_POLYNOMIAL,
		OMURTAG_SPECIFIC_POLYNOMIAL,
		true
	);

	return true;
}


bool TestPopulist::OmurtagMatrixTest() const
{
	GenericOnePopulationTest
	(
		OMURTAG_RATE,
		OMURTAG_EFFICACY,
		OMURTAG_PARAMETER,
		OMURTAG_PARAMETER_MATRIX,
		OMURTAG_SPECIFIC_MATRIX,
		true
	);
	
	return true;
}

bool TestPopulist::TestResponseCurveSingle(bool b_refractive) const
{
	for (Index i = 0;	i < NUMBER_RESPONSE_CURVE_POINTS; i++)
	{
		PopulistSpecificParameter par_spec;
		Pop_Network		network;
		Rate			rate_exc;
		NodeId			id;

		ResponseCurveSingleNetwork<SingleInputZeroLeakEquations>
		(
			&network,
			&par_spec,
			b_refractive,
			false,
			false,
			&rate_exc,
			&id
		);

		string handlername;
		string logname;				

		ResponseSingleRunParameter
		(
			b_refractive,
			false,
			false,
			false,
			i,
			handlername,
			logname,
			&par_spec,
			RESPONSE_CURVE_SINGLE_NBINS
		); 
	

		RootReportHandler 
			handler
			(
				handlername.c_str(),
				ONSCREEN,
				INFILE,
				RESPONSE_CANVAS
			);

		SimulationRunParameter
		par_run
		(
			handler,
			RESPONSE_CURVE_MAX_ITER,
			RESPONSE_CURVE_T_BEGIN,
			RESPONSE_CURVE_T_END,
			RESPONSE_CURVE_T_REPORT,
			RESPONSE_CURVE_T_REPORT,
			RESPONSE_CURVE_T_NETWORK,
			logname
		);


		SinglePopulationInput inp;
		inp =  CovertMuSigma(MU[i], SIGMA, PARAMETER_NEURON);
		// Set the rate variable:
		rate_exc = inp._rate;
		// Seek out the connection from the rate algorithm and set the weight
		PopulationConnection con(1,inp._h);

		Pop_Network::predecessor_iterator iter = network.begin(id);
		iter.SetWeight(con);	

		bool b_configure =
 			network.ConfigureSimulation(par_run);

		if (! b_configure)
			return false;

		par_run.Handler().AddNodeToCanvas(id);

		bool b_evolve = network.Evolve();
		if (! b_evolve)
			return false;
	}

	return true;
}

bool TestPopulist::TestResponseCurveDouble(bool b_refractive) const
{


	for (Index i = 0;	i < NUMBER_RESPONSE_CURVE_POINTS; i++)
	{
		Pop_Network					network;
		PopulistSpecificParameter	par_spec;
		Rate						input_rate_exc;
		Rate						input_rate_inh;
		NodeId						id;

		ResponseCurveDoubleNetwork<NumericalZeroLeakEquations>
		(
			&network,
			&par_spec,
			b_refractive,
			false,
			false,
			&input_rate_exc,
			&input_rate_inh,
			&id
		);
	

		string handlername;
		string logname;				

		ResponseSingleRunParameter
		(
			b_refractive,
			false,
			false,
			true,
			i,
			handlername,
			logname,
			&par_spec,
			RESPONSE_CURVE_SINGLE_NBINS
		); 	

		RootReportHandler 
			handler
			(
				handlername.c_str(),
				ONSCREEN,
				INFILE,
				RESPONSE_CANVAS
			);
	
		SimulationRunParameter
		par_run
		(
			handler,
			RESPONSE_CURVE_MAX_ITER,
			RESPONSE_CURVE_T_BEGIN,
			RESPONSE_CURVE_T_END,
			RESPONSE_CURVE_T_REPORT,
			RESPONSE_CURVE_T_REPORT,
			RESPONSE_CURVE_T_NETWORK,
			logname
		);

		par_run.Handler().AddNodeToCanvas(id);

		if ( network.begin(id) != network.end(id) ){
			Pop_Network::predecessor_iterator iter = network.begin(id);

			// precise values don't matter, as long as they reproduce correct mu and sigma
			double h_e = 1;
			double N_e = (MU[i] + SIGMA*SIGMA)/(2*PARAMETER_NEURON._tau);
			PopulationConnection con_e(N_e,h_e);
			iter.SetWeight(con_e);
			input_rate_exc = 1;

			double h_i = -1;
			double N_i = (SIGMA*SIGMA -MU[i])/(2*PARAMETER_NEURON._tau);
			PopulationConnection  con_i(N_i,h_i);
			iter++;
			iter.SetWeight(con_i);
			input_rate_inh = 1;
		}

		bool b_configure =
			network.ConfigureSimulation(par_run);

		if (! b_configure)
			return false;

		bool b_evolve = network.Evolve();

		if (! b_evolve)
			return false;
	}	
	return true;
}

void TestPopulist::ResponseSpecificParameter
(
	bool						b_fit,
	bool						b_short,
	bool						b_double,
	PopulistSpecificParameter*	p_spec,
	Number						n_dif,
	double						lim_dif
 ) const
{

	Number n_bins;

	if (b_double)
	{
		n_bins = RESPONSE_CURVE_DOUBLE_NBINS;
	}
	else
	{
		n_bins = RESPONSE_CURVE_SINGLE_NBINS;
	}

	const AbstractRateComputation* p_rate;
	if ( b_fit )
		p_rate = &FIT_RATE_COMPUTATION;
	else
		p_rate = &INTEGRAL_RATE_COMPUTATION;

	const AbstractCirculantSolver*    p_circulant;
	const AbstractNonCirculantSolver* p_non_circulant; 

	if (b_short)
	{
		p_circulant     = &POLYNOMIAL_CIRCULANT;
		p_non_circulant = &LIMITED_NON_CIRCULANT;
	}
	else
	{
		p_circulant = 0;
		p_non_circulant = 0;
	}



	PopulistSpecificParameter
		par_spec
		(
			RESPONSE_CURVE_V_MIN,
			n_bins,
			RESPONSE_CURVE_NADD,
			RESPONSE_CURVE_INITIAL_DENSITY,
			RESPONSE_CURVE_EXPANSION_FACTOR,
			"NumericalZeroLeakEquations"
		);

	*p_spec = par_spec;
}

void TestPopulist::ResponseSingleRunParameter
(
	bool							b_refractive,
	bool							b_fit,
	bool							b_short,
	bool							b_double,
	Index							i,
	string&							str_handler,
	string&							str_log,
	PopulistSpecificParameter*		p_par_spec,
	Number							n_bins
 ) const
{
	string str = RESPONSE_CURVE_ROOT_FILE;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep(".");
	tokenizer tokens(str, sep);
	tokenizer::iterator tok_iter = tokens.begin();

	string str_ref = b_refractive ? "_refractive" : "_nonrefractive";
	string str_fit = b_fit ? "_fitrate" : "_intrate";
	string str_mod = b_double ? "_double" : "_single";
	string str_sho = b_short ? "_shorttime" : "_fulltime"; 
	// first generate the appropriate name
	ostringstream stream;
	stream << *tok_iter;
	stream << str_ref;
	stream << str_fit;
	stream << str_sho;
	stream << str_mod;
	stream << "_" << n_bins << "_";
	stream << i;
	stream << ".root";
	str_handler = stream.str();

	ostringstream stream_log;
	stream_log << *tok_iter;
	stream_log << str_fit;
	stream_log << str_sho;
	stream_log << str_mod;
	stream_log << "_" << n_bins << "_";
	stream_log << i;
	stream_log << ".log";

	str_log = stream_log.str();
}


bool TestPopulist::ResponseCurveExample() const
{
	//This test runs a Wilson-Cowan node with a LIF gain function as a sigmoid. Its steady state is the same steady state output firing rate
	// is the same of that of a group of LIF neurons. It can be used to estimate the output of populations that run a PopulistAlgorithm.

	// Note we now need an OU_Network instead of a D_Network
	OU_Network network;

	Potential sigma = 2e-3;
	Potential mu    = 20e-3;

	Time tau = PARAMETER_NEURON._tau; 
	Rate nu = mu*mu/(sigma*sigma*tau);
	Rate J = sigma*sigma/mu;

	OU_Connection 
		con
		(
			1,
			J
		);

	// Define a node with a fixed output rate
	OU_RateAlgorithm rate_alg(nu);
	NodeId id_rate = network.AddNode(rate_alg,EXCITATORY);

	// Define the receiving node 
	OU_Algorithm algorithm_exc(PARAMETER_NEURON);
	NodeId id = network.AddNode(algorithm_exc,EXCITATORY);

	// connect the two nodes
	network.MakeFirstInputOfSecond(id_rate,id,con);

	// define a handler to store the simulation results
	RootReportHandler 
		handler
		(
			"test/ouresponse.root",	// file where the simulation results are written
			TEST_RESULTS_ON_SCREEN,	// do not display on screen
			true					// write into file
		);


	SimulationRunParameter
		par_run
		(
			handler,				// the handler object
			1000000,				// maximum number of iterations
			0,						// start time of simulation
			1.0,					// end time of simulation
			1e-4,					// report time
			1e-4,					// update time
			1e-5,					// network step time
			"test/ouresponse.log"   // log file name
		);

	bool b_configure = network.ConfigureSimulation(par_run);
	if (! b_configure)
		return false;

	bool b_evolve = network.Evolve();
	if (! b_evolve)
		return false;
	return true;
}

bool TestPopulist::PrintResponseCurve() const
{	ofstream file("test/responsecurvegraph.txt");
	double mu_begin = 10e-3;
	const int NR_SIGMA = 4;
	const double sigma[NR_SIGMA] = {1e-3,2e-3,5e-3,7e-3};
	double mu_end = 30e-3;

	int n_steps = 10000;
	for (int i_ref = 0; i_ref < 2; i_ref++) 
		for (int i_sigma = 0; i_sigma < NR_SIGMA; i_sigma++){
			for (int i = 0; i < n_steps; i++){
			double delta_mu = (mu_end - mu_begin)/(n_steps - 1);
			double mu = mu_begin + i*delta_mu; 
			ResponseParameterBrunel par_brun;
			par_brun.tau			= PARAMETER_NEURON._tau;
			par_brun.mu				= mu;
			par_brun.sigma			= sigma[i_sigma];
			par_brun.tau_refractive = i_ref*PARAMETER_NEURON._tau_refractive;
			par_brun.theta			= PARAMETER_NEURON._theta;
			par_brun.V_reset		= PARAMETER_NEURON._V_reset;

			file << mu << " " << sigma[i_sigma] << " " << par_brun.tau_refractive << " " << ResponseFunction(par_brun) << endl;
		    }
		}
	return true;
}


namespace {

	DynamicLib::Rate rate_function_linear(DynamicLib::Time time)
	{
		return time;
	}
}

bool TestPopulist::ScalarProductTest() const
{
	RateFunctor<OU_Connection> ra(rate_function_linear);

	OU_DynamicNode node_e(ra,EXCITATORY);
	OU_DynamicNode node_i(ra,INHIBITORY);

	//Create a connection list by hand

	vector<OU_DynamicNode::connection> vec_con;

	OU_Connection c_e, c_i;
	c_e._number_of_connections = 1.0;
	c_e._efficacy			   = 2.0;

	c_i._number_of_connections = 0.5;
	c_i._efficacy			   = 3.0;

	OU_DynamicNode::connection con_1;
	con_1.first  = &node_e;
	con_1.second = c_e;

	OU_DynamicNode::connection con_2;
	con_2.first  = &node_i;
	con_2.second = c_i;

	// this vector is a connection list as they are present in OU_DynamicNode s
	vec_con.push_back(con_1);
	vec_con.push_back(con_2);

	node_e.SetValue(10.0);
	node_i.SetValue(20.0);

	MuSigmaScalarProduct product;
	// the ugly exposition to the innards of the vector is normally done behind the veils of SparseNode
	MuSigmaScalarProduct::predecessor_iterator iter_begin(&vec_con[0]);
	MuSigmaScalarProduct::predecessor_iterator iter_end(&vec_con[vec_con.size()-1] + 1);
	MuSigma musig =
		product.Evaluate
		(
			iter_begin,
			iter_end,
			1.0
		);

	return true;
}

bool TestPopulist::AEIFIntegratorTest() const
{
	// set adaptation to 0
	AEIFParameter param;
	param._a = param._b = 0;

	// add 10% to the interval [V_L, V_th) and initialize a vector
	Potential V_min = 1.1*param._E_l;
	Potential V_max = 0.9*param._V_t;

	Number n_steps = 100;
	
	Potential Delta = (V_max - V_min)/(n_steps - 1);
	vector<double> v_init(n_steps);

	for ( Index i = 0; i < n_steps; i++)
		v_init[i] = V_min + i*Delta; 

	// Now integrate every curve for 2 ms

	for ( Index j = 0; j < n_steps; j++)
	{
		vector<double> vec(param.StateDimension());
		
		vec[V] = v_init[j];
		vec[W] = 0.0;

		DVIntegrator<AEIFParameter> 
			integ
			(
				100000,						//<! maximum number of integrations							
				vec,						//<! initial state
				0.01,						//<! initial time step
				0.0,						//<! initial time
				Precision(0.01,0.01),			//<! precsion
				aeifdydt,					//<! dydt
				0,							//<! Jacobian, if available
				gsl_odeiv_step_rkf45		//<! gsl integrator object
			);
	}
	return true;
}

namespace {

	DynamicLib::Rate JumpFunction(DynamicLib::Time t)
	{
		return (t < 0.3 || t > 0.6 )? 6.0 : 8.0;
	}
}
bool TestPopulist::OneDMTest() const
{
	// set a nominal input in terms of a and b
	Pop_RateFunctor jump(JumpFunction);

	PopulationConnection con(1.0,1.0);

	Pop_Network net;
	NodeId id_a = net.AddNode(jump,EXCITATORY);

	Potential g_max = 100;
	AdaptationParameter 
		par_adapt
		(
			110e-3,	// adaptation time constant
			14.48,	// adaptation jump value
			g_max	// maximum g value
		);

	PopulistSpecificParameter 
		par_spec
		(
			0.0,								//!< minimum potential of the grid, (typically negative or below the reversal potential
			100,								//!< initial number of bins
			1,									//!< number of bins that is after one zero-leak evaluation
			InitialDensityParameter(0.0,0.0),	//!< gaussian (or delta-peak) initial density profile
			1.1,									//!< expansion factor
			"OneDMZeroLeakEquations"
		);

	PopulationParameter dummy;
	OneDMParameter 
		par_onedm
		(
			dummy,
			par_adapt,
			par_spec
		);

	OneDMAlgorithm<PopulationConnection> 
		alg_onedm
		(
			par_onedm
		);

	NodeId id_one = net.AddNode(alg_onedm,EXCITATORY);
	net.MakeFirstInputOfSecond(id_a,id_one,con);

	
	// define a handler to store the simulation results


	const CanvasParameter 
		ONEDM_CANVAS 
		(
			0.0,
			1.0,
			0.0,
			70.0,
			0.0,
			70.0,
			0.0,
			0.5
		);

	RootReportHandler 
		handler
		(
			"test/onedm_10000.root",		// file where the simulation results are written
			TEST_RESULTS_ON_SCREEN,			// do not display on screen
			true,							// write into file
			ONEDM_CANVAS
		);

	handler.AddNodeToCanvas(id_one);

	SimulationRunParameter
		par_run
		(
			handler,				// the handler object
			1000000,				// maximum number of iterations
			0,						// start time of simulation
			1.0,					// end time of simulation
			1e-2,					// report time
			1e-2,					// update time
			1e-4,					// network step time
			"test/onedmresponse.log"// log file name
		);

	net.ConfigureSimulation(par_run);

	net.Evolve();

	return true;
}

bool TestPopulist::HazardFunctionTest() const
{
	// See if the numerical values of the Hazard function are in the right ball park. You can read them from
	// the following file:

	ofstream st("test/hazard.txt");

	if (! st)
		return false;

	float a = 6.91423056F;	// Hz
	float b = 0.132995626F; // [nS]^-1
//	float tau = 110.0F;		// ms // given in paper but not used here
	float q = 14.48F;		// nS

	AdaptiveHazard hazard(a,b);

	int n_steps = 100;
	float g_max = 10*q;

	float delta = g_max/(n_steps - 1);
	for (int i = 0; i < n_steps; i++)
	{	
		float g = i*delta;
		st << i << " " << g << " " << hazard(g) << " " << hazard(g-q) << endl;
	}

	return true;
}

bool TestPopulist::StreamPopulationAlgorithmIn() const
{
	PopulistParameter par = TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER_POP;
	ofstream st("test/streampop.alg");
	
	PopulationAlgorithm alg(par);
	alg.SetName("goeroeboeroe met toebehoor");
	st << alg;

	// now try the same with a copy
	ofstream stc("test/streampopcopy.alg");
	boost::shared_ptr<PopulationAlgorithm> p_alg;
	p_alg = boost::shared_ptr<PopulationAlgorithm>(alg.Clone());
	p_alg->ToStream(stc);

	return true;
}

bool TestPopulist::StreamPopulationAlgorithmOut() const
{
	ifstream st("test/streampop.alg");

	PopulationAlgorithm alg(st);

	return true;
}

bool TestPopulist::BuildPopulationAlgorithm() const
{
	ifstream st("test/streampop.alg");

	AlgorithmBuilder<PopulationConnection> build;
	boost::shared_ptr< AbstractAlgorithm<PopulationConnection> > p_alg = build.Build(st);

	return true;
}

// The next function is in for documentation purposes, it will be picked up by doxyegen

bool TestPopulist::BalancedExample() const
{

	OrnsteinUhlenbeckParameter par;
	par._theta      = 20e-3; // (V)
	par._tau        = 10e-3; // (V)
	par._V_reversal = 0.0;
	par._V_reset    = 0.0;

	// Set mu and sigma
	double mu    = 20e-3;    // (V)
	double sigma = 2e-3;     // (V)

	// In order to approximate a diffusion process set a small value for input
	// weights (small relative to theta).
	//
	double J= 0.01* par._theta;

	// Now convert mu and sigma to input rates of an excitatory and inhibitory
	// population.
	//
	double nu_e = (J*mu + sigma*sigma)/(2*J*J*par._tau);
	double nu_i = (sigma*sigma - J*mu)/(2*J*J*par._tau);
//
// some parameters specific to the algorithm
//
	double V_min  = -10e-3;
	int n_bins    = 500;
	int n_add     = 1;
	double f_exp  = 1.1;
 
	// delta-peak at the reversal potential
	InitialDensityParameter density(par._V_reversal,0);

	PopulistSpecificParameter 
	par_spec
	(
		V_min,
		n_bins,
		n_add,
		density,
		f_exp
	);

	//
	// Now create the network
	//

	Pop_Network			net;
	PopulistParameter	par_pop(par,par_spec);
	PopulationAlgorithm alg_pop(par_pop);
	NodeId id_pop  =	net.AddNode(alg_pop,EXCITATORY);
	//
	// Create input populations and add them to the network
	//
	Pop_RateAlgorithm alg_rate_exc(nu_e);
	NodeId id_e = net.AddNode(alg_rate_exc,EXCITATORY);
	OrnsteinUhlenbeckConnection con_e(1,J);
	Pop_RateAlgorithm alg_rate_inh(nu_i);
	NodeId id_i = net.AddNode(alg_rate_inh,INHIBITORY);
	OrnsteinUhlenbeckConnection con_i(1,-J);
	//
	net.MakeFirstInputOfSecond(id_e,id_pop,con_e);
	net.MakeFirstInputOfSecond(id_i,id_pop,con_i);
	//
	RootReportHandler handler("data.root",TEST_RESULTS_ON_SCREEN,true);
	handler.SetFrequencyRange(0,40);
	handler.SetDensityRange(-0.01,200);
	handler.SetTimeRange(0,0.3);
	handler.SetPotentialRange(-0.001,0.020);
	handler.AddNodeToCanvas(id_pop);
	//
	// Configure the simulation
	//
	SimulationRunParameter 
	par_run
	(
		handler,
		100000,
		0.,
		0.3,
		1e-2,
		1e-2,
		1e-2,
		"simulation.log"
	);

	net.ConfigureSimulation(par_run);
	net.Evolve();

	return true;
	// end balanced example
}

bool TestPopulist::ZeroLeakBuilderTest() const
{
		Number						n_bins;
		valarray<Potential>			array_state;
		Potential					checksum;
		SpecialBins					bins;
		PopulationParameter			par_pop;	
		PopulistSpecificParameter	par_spec;	
		Potential					delta_v;

	ZeroLeakBuilder 
		build
		(
			n_bins,
			array_state,
			checksum,
			bins,
			par_pop,
			par_spec,
			delta_v
		);


	boost::shared_ptr<AbstractZeroLeakEquations> p_zl = build.GenerateZeroLeakEquations("LIFZeroLeakEquations","CirculantSolver","NonCirculantSolver");

	return true;
}

bool TestPopulist::StreamOUAlgorithmIn() const
{
	ifstream ifst("test/ou.alg");
	if ( !ifst )
		return false;

	OU_Algorithm alg(ifst);

	ofstream ofst("test/newou.alg");
	alg.ToStream(ofst);

	return true;
}

bool TestPopulist::StreamOUAlgorithmOut	() const
{
	OU_Algorithm alg(TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER);
	alg.SetName("bomba zomba");
	ofstream ofst("test/ou.alg");

	if (! ofst)
		return false;
	alg.ToStream(ofst);

	return true;
}

bool TestPopulist::BuildOUAlgorithm	() const
{	
	ifstream st("test/ou.alg");

	AlgorithmBuilder<PopulationConnection> build;
	boost::shared_ptr< AbstractAlgorithm<PopulationConnection> > p_alg = build.Build(st);

	ofstream ofst("test/buildou.alg");
	p_alg->ToStream(ofst);

	return true;
}



bool TestPopulist::InputConvertorTest() const
{
	Rate rate_birst = 800;
	OU_RateAlgorithm input(rate_birst);
	OU_DynamicNode node_burst(input,EXCITATORY_BURST);
	node_burst.SetValue(rate_birst);

	Rate rate_diffusion = 1000;
	OU_RateAlgorithm diffusion_exc(rate_diffusion);
	OU_DynamicNode node_difexc(diffusion_exc,EXCITATORY);
	node_difexc.SetValue(rate_diffusion);

	OU_RateAlgorithm diffusion_inh(rate_diffusion);
	OU_DynamicNode node_difinh(diffusion_inh,EXCITATORY);
	node_difinh.SetValue(rate_diffusion);

	Efficacy eff_single = 0.03;
	Efficacy eff_difusion = 0.001;

	PopulationParameter par_pop;
	par_pop._tau			= 20e-3; // only value that really matters
	par_pop._tau_refractive = 0.0;
	par_pop._V_reset		= 0.0;
	par_pop._V_reversal		= 0.0;
	par_pop._theta			= 20e-3;

	SpecialBins bins;
	PopulistSpecificParameter par_spec;
	Number n_bins;
	Potential delta_v = 0.01;

	LIFConvertor
		conv
		(
			bins,
			par_pop,
			par_spec,
			delta_v,
			n_bins
		);

	typedef pair<AbstractSparseNode<OU_DynamicNode::ActivityType,PopulationConnection>*,PopulationConnection>	Connection;

	PopulationConnection popcon_exc(1,eff_single);
	PopulationConnection popcon_dife(1,eff_difusion);
	PopulationConnection popcon_difi(1,-eff_difusion);

	Connection con_exc;
	con_exc.first  = &node_burst;
	con_exc.second = popcon_exc;
	Connection con_dife;
	con_dife.first  = &node_difexc;
	con_dife.second = popcon_dife;
	Connection con_difi;
	con_difi.first  = &node_difinh;
	con_difi.second = popcon_difi;

	vector<Connection> vec_con;
//	vec_con.push_back(con_exc);
	vec_con.push_back(con_dife);
	vec_con.push_back(con_difi);
	
	LIFConvertor::predecessor_iterator iter_begin(&(*vec_con.begin()));
	LIFConvertor::predecessor_iterator iter_end(&(* vec_con.begin() ) + vec_con.size());

	conv.SortConnectionvector(iter_begin,iter_end);
	conv.AdaptParameters();

	return true;
}


     bool TestPopulist::GenericOnePopulationTest
		(
			Rate								input_rate,
			PopulistLib::Efficacy				input_weight,
			const PopulationParameter&			par_pop,
			const SimulationRunParameter&		par_run,
			const PopulistSpecificParameter&	par_spec,
			bool								b_log
		) const

	{
		return GenericTwoPopulationTest
		(	
			input_rate,
			input_weight,
			0,
			0,
			par_pop,
			par_run,
			par_spec,
			b_log
		);
	}

	bool TestPopulist::GenericTwoPopulationTest
	(
		Rate								input_rate_1,
		PopulistLib::Efficacy				input_weight_1,
		Rate								input_rate_2,
		PopulistLib::Efficacy				input_weight_2,
		const PopulationParameter&			par_population,
		const SimulationRunParameter&		par_run,
		const PopulistSpecificParameter&	par_spec,
		bool								b_log
	) const
	{
		//TODO: for the time being assume: one population bursting, two populations diffusion. Must be remedied.
		bool burst = (input_weight_2 == 0 )? true : false;
		// This allows for testing the adapting bin parameters
		Pop_Network network;

		// Define an rate population
		RateAlgorithm < PopulationConnection > rate_input_1 (input_rate_1);
		RateAlgorithm < PopulationConnection > rate_input_2 (input_rate_2);

		NodeId id_rate_node_1, id_rate_node_2;
	
		if ( input_weight_1 >= 0 ){
			id_rate_node_1 = 
				network.AddNode 
				(
					rate_input_1,
					burst ? EXCITATORY_BURST : EXCITATORY
				);
		}
		else {

			id_rate_node_1 = 
				network.AddNode 
				(
					rate_input_1,
					burst ? INHIBITORY_BURST : INHIBITORY
				);
		}

		if ( input_weight_2 >= 0 ){
			id_rate_node_2 = 
				network.AddNode 
				(
					rate_input_2,
				burst ? EXCITATORY_BURST : EXCITATORY
				);
			}
		else {

			id_rate_node_2 = 
				network.AddNode 
				(
					rate_input_2,
					burst ? INHIBITORY_BURST : INHIBITORY
				);
		}

		// Define the node
		PopulationAlgorithm_<PopulationConnection> 
			the_algorithm (PopulistParameter(par_population,par_spec));

		NodeId id_the_node = 
			network.AddNode 
			(
				the_algorithm,
				EXCITATORY
			);

		PopulationConnection connection_1 (1, input_weight_1);
		PopulationConnection connection_2 (1, input_weight_2);

		// add the first rate node, no matter what
		network.MakeFirstInputOfSecond
		(
			id_rate_node_1, 
			id_the_node, 
			connection_1
		);

		// but the 2nd only if it is non-zero
		if ( input_weight_2 != 0 )
			network.MakeFirstInputOfSecond
			(
				id_rate_node_2,
				id_the_node,
				connection_2
			);

		SimulationRunParameter parameter_run_current = par_run;
		parameter_run_current.Handler().AddNodeToCanvas
		(
			id_the_node
		);

		bool b_configure =
			network.ConfigureSimulation (par_run);

		if (b_configure)
		{
			bool b_return;
			Timer time;
			time.SecondsSinceLastCall();
			b_return =  network.Evolve ();
		}
		else
			return false;

		return true;
	}

	bool TestPopulist::RootFileInterpreterTest() const
	{
		// This test is here, because there are no relevant tests in the Unit test suite at the moment. It may
		// be a good idea to move it there later.

		TFile file("test/omurtag.root");
		RootFileInterpreter inter(file);

		vector<Time> vec = inter.StateTimes(NodeId(3));
		Time t1 = vec[0];
		Time t2 = vec[1];
		
		TGraph* p1 = inter.GetStateGraph(NodeId(3),t1);
		TGraph* p2 = inter.GetStateGraph(NodeId(3),t2);

		if (!p1 || ! p2)
			return false;

		return true;
	}

	string TestPopulist::ResponseCurveFileName
	(
		bool b_double,
		bool b_refractive,
		Index i
	) const
	{
		string curve;
		string dummy;

		PopulistSpecificParameter par_spec;
		this->ResponseSingleRunParameter
		(
			b_refractive,
			false,
			false,
			b_double,
			i,
			curve,
			dummy,
			&par_spec,
			RESPONSE_CURVE_SINGLE_NBINS
		);

		return curve;
	}

namespace { 
	void RetrieveRatePrediction(double sigma, double tau_ref, vector<double>* p_vec_mu, vector<double>* p_rate){
		p_vec_mu->clear();
		p_rate->clear();

		double m, rate, sig, t_ref;
		ifstream datafile("test/responsecurvegraph.txt");

		if (!datafile)
			throw PopulistException("Couldn't open response curve data file.Did you run all test functions?");

		while(datafile){
			datafile >> m >> sig >> t_ref >> rate;
			if( NumtoolsLib::IsApproximatelyEqualTo(sig,sigma,1e-6) && NumtoolsLib::IsApproximatelyEqualTo(tau_ref,t_ref,1e-6) ){
				p_vec_mu->push_back(m);
				p_rate->push_back(rate);
			}
		}
	}

	void RetrieveRateMeasurement(bool b_double, bool b_refractive, vector<double>* p_vec_mu, vector<double>* p_rate){
		boost::shared_ptr<ostream> p;
		TestPopulist test(p);

		p_vec_mu->clear();
		p_rate->clear();

		for (Index i = 0; i < NUMBER_RESPONSE_CURVE_POINTS; i++){
			string str = test.ResponseCurveFileName(b_double,b_refractive,i);
			TFile file(str.c_str());
			RootFileInterpreter inter(file);

			p_vec_mu->push_back(MU[i]);
			if (b_double)
				p_rate->push_back(inter.ExtractFiringRate(NodeId(3),0.5*RESPONSE_CURVE_T_END,RESPONSE_CURVE_T_END));
			else
				p_rate->push_back(inter.ExtractFiringRate(NodeId(1),0.5*RESPONSE_CURVE_T_END,RESPONSE_CURVE_T_END));
		}
	}

	void Graph1(){
		// example 1: standard one population denst with density shown at two different points in time
		gROOT->SetStyle("Plain");
		gROOT->SetBatch(1);
		gStyle->SetOptStat(0);

		TFile file("test/omurtag.root");

		// get the first and the last state file
		RootFileInterpreter inter(file);

		vector<Time> vec = inter.StateTimes(NodeId(3));
		//yes! naked pointers, due to the peculiairities of ROOT's ownership model	
		TGraph* p_state  = inter.GetStateGraph(NodeId(3),0.0);
		TGraph* p_state2 = inter.GetStateGraph(NodeId(3),1.0);

		TGraph* p_rate =   inter.GetRateGraph(NodeId(3)); 
		if (! p_rate)
		  throw PopulistException("No rate graph in simulation file");

		TH2F hist2("h","Population density",500,0,1.0,500,0,4.0);
		hist2.SetXTitle("normalized membrane potential");
		hist2.SetYTitle("density");
		TCanvas c1;
		c1.Divide(1,2);
		c1.cd(1);
		TSVG your_ps("test/populistlib_test_omurtagstate.svg");

		your_ps.Range(15,12);
		hist2.Draw();
		p_state->Draw("L");

		c1.cd(2);
		hist2.Draw();
		p_state2->Draw("L");

		your_ps.Close();
		file.Close();
	}

	void Graph2(){

		// graph of the population response
		TFile file2("test/omurtag.root");

		RootFileInterpreter inter(file2);
		TGraph* p_gr = inter.GetRateGraph(NodeId(3)); // this is the excitatory population

		gROOT->SetStyle("Plain");
		gROOT->SetBatch(1);
		gStyle->SetOptStat(0);


		TH2F hist2("h","Population rate",500,0,0.3,500,0,20.0);
		hist2.SetXTitle("t (s)");
		hist2.SetYTitle("spikes/s");
		TCanvas c1;

		TSVG your_ps("test/populistlib_test_omurtagresponse.svg");

		your_ps.Range(15,12);
		hist2.Draw();
		p_gr->Draw("L");

		your_ps.Close();
		file2.Close();
	}

	void Graph3(){
		// example 1: coupled excitatory-inhibitory population with inhibitory input: firing rates
		gROOT->SetStyle("Plain");
		gROOT->SetBatch(1);
		gStyle->SetOptStat(0);

		TFile file("test/twopoptest.root");

		RootFileInterpreter inter(file);
		TGraph* p_rate_e =  inter.GetRateGraph(NodeId(2)); //yes! naked pointers, due to the peculiairities of ROOT's ownership model
		TGraph* p_rate_i =  inter.GetRateGraph(NodeId(3)); //yes! naked pointers, due to the peculiairities of ROOT's ownership model

		TH2F hist2("h","Population density",500,0,0.05,500,0,8.0);
		hist2.SetXTitle("t(s)");
		hist2.SetYTitle("spikes/s");
		TCanvas c1;
		c1.Divide(1,2);
		c1.cd(1);
		TSVG your_ps("test/populistlib_test_twopopresponse.svg");

		your_ps.Range(15,12);
		hist2.Draw();
		p_rate_e->Draw("L");

		c1.cd(2);
		hist2.Draw();
		p_rate_i->Draw("L");

		your_ps.Close();
		file.Close();
	}

	void Graph4(){
		// example 2: population density of the excitatory-inhibitory population circuit
		gROOT->SetStyle("Plain");
		gROOT->SetBatch(1);
		gStyle->SetOptStat(0);

		TFile file("test/twopoptest.root");

		RootFileInterpreter inter(file);

		TGraph* p_state_e =  inter.GetStateGraph(NodeId(2),0.05); //yes! naked pointers, due to the peculiairities of ROOT's ownership model
		TGraph* p_state_i =  inter.GetStateGraph(NodeId(3),0.05); //yes! naked pointers, due to the peculiairities of ROOT's ownership model

		TH2F hist2("h","Population density",500,0,0.020,500,0,250.0);
		hist2.SetXTitle("t(s)");
		hist2.SetYTitle("spikes/s");
		TCanvas c1;
		c1.Divide(1,2);
		c1.cd(1);
		TSVG your_ps("test/populistlib_test_twopopstate.svg");

		your_ps.Range(15,12);
		hist2.Draw();
		p_state_e->Draw("L");

		c1.cd(2);
		hist2.Draw();
		p_state_i->Draw("L");

		your_ps.Close();
		file.Close();
	}



	void Graph5(){
		// response of single population with and without refraction
		gROOT->SetStyle("Plain");
		gROOT->SetBatch(1);
		gStyle->SetOptStat(0);

		TCanvas c1;

		TFile file("test/omurtag_refractive.root");
		TGraph* p_ref = (TGraph*)file.Get("rate_3");
		file.Close();

		TFile file_zero("test/omurtag_refractive_zero.root");
		TGraph* p_zero = (TGraph*)file_zero.Get("rate_3");
		file_zero.Close();

		TH2F hist("h","response w vs wo refraction",500,0.,0.3,500,0.,20.);

		TSVG your_ps("test/refractive.svg");
		your_ps.Range(15,12);

		hist.Draw();
		p_zero->Draw("L");
		p_ref->SetLineColor(2);
		p_ref->Draw("L");

		your_ps.Close();
	}

	void Graph6(){
		// response of single population with and without refraction
		gROOT->SetStyle("Plain");
		gROOT->SetBatch(1);
		gStyle->SetOptStat(0);

		TCanvas c1;

		TFile file("test/omurtag_numerical_refractive.root");
		TGraph* p_ref = (TGraph*)file.Get("rate_3");
		file.Close();

		TFile file_zero("test/omurtag_numerical.root");
		TGraph* p_zero = (TGraph*)file_zero.Get("rate_3");
		file_zero.Close();

		TH2F hist("h","response w vs wo refraction (num)",500,0.,0.3,500,0.,20.);

		TSVG your_ps("test/numerical_refractive.svg");
		your_ps.Range(15,12);

		hist.Draw();
		p_zero->Draw("L");
		p_ref->SetLineColor(2);
		p_ref->Draw("L");

		your_ps.Close();
	}

	void Graph7(){
		gROOT->SetStyle("Plain");
		gROOT->SetBatch(1);
		gStyle->SetOptStat(0);

		TCanvas c1;
		vector<double> mu;
		vector<double> rate;

		double sigma   = 2.0e-3;
		double tau     = 0.0;

		RetrieveRatePrediction(sigma, tau, &mu,&rate);
		TGraph g_non_ref(mu.size(),&mu[0],&rate[0]);
		

		vector<double> mu_ref;
		vector<double> rate_ref;

		double sigma_ref = 2.0e-3;
		double tau_ref   = 4.0e-3;

		RetrieveRatePrediction(sigma_ref,tau_ref,&mu_ref,&rate_ref);
		TGraph g_ref(mu_ref.size(),&mu_ref[0], &rate_ref[0]);
		g_ref.SetLineColor(2);

		vector<double> mu_measured;
		vector<double> rate_measured;
		RetrieveRateMeasurement(true,false,&mu_measured,&rate_measured);
		TGraph g_mes(mu_measured.size(),&mu_measured[0],&rate_measured[0]);
		g_mes.SetMarkerSize(1.0);
		g_mes.SetMarkerStyle(2);

		vector<double> mu_measured_ref;
		vector<double> rate_measured_ref;
		RetrieveRateMeasurement(true,true,&mu_measured_ref,&rate_measured_ref);
		TGraph g_mes_ref(mu_measured_ref.size(),&mu_measured_ref[0],&rate_measured_ref[0]);
		g_mes_ref.SetMarkerSize(1.0);
		g_mes_ref.SetMarkerStyle(3);
		g_mes_ref.SetMarkerColor(2);


		TH2F hist("h","analytic response vs simulations", 500, 10e-3, 25e-3, 500, 0., 35.);
		TSVG your_ps("test/response_curver.svg");
		your_ps.Range(15,12);
		hist.Draw();
		g_ref.Draw("L");
		g_non_ref.Draw("L");
		g_mes.Draw("P");
		g_mes_ref.Draw("P");
		your_ps.Close();
	}
}


void TestPopulist::ProcessResults()
{
/*	Graph1();
	Graph2();
	Graph3();
	Graph4();
	Graph5();
	Graph6();*/
	Graph7();
}
