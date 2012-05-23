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
#ifndef _CODE_LIBS_POPULISTLIB_TESTOMURTAGDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TESTOMURTAGDEFINITIONS_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "FitRateComputation.h"
#include "InterpolationRebinner.h"
#include "LimitedNonCirculant.h"
#include "MatrixNonCirculant.h"
#include "PolynomialCirculant.h"
#include "PopulistSpecificParameter.h"
#include "TestDefinitions.h"
#include "TestInhibitionDefinitions.h"

using DynamicLib::RootReportHandler;

namespace PopulistLib {

	// OMURTAGTEST

	const Rate      OMURTAG_RATE                   = 800;  // Hz							
	const Efficacy  OMURTAG_EFFICACY               = 0.03; // rescaled potential

	const double    OMURTAG_EXPANSION_FACTOR		= 1.1;  // maximum expansion before rebinning
	const Number    OMURTAG_NUMBER_OF_BINS_TO_ADD	= 1;    // add one bin at a time


	const Number    OMURTAG_NUMBER_INITIAL_BINS		= 330;  // number of bins at start of simulation
	const Potential OMURTAG_V_MIN					= -0.1;

	const InitialDensityParameter 
		OMURTAG_INITIAL_DENSITY
		(
			0,
			0
		);

	const PopulationParameter 
		OMURTAG_PARAMETER
		(
			1,    // theta
			0,    // V_reset
			0,    // V_reversal
			0,    // no refractive period
			50e-3 // 50 ms or 20 s^-1
		);


	const Time	OMURTAG_T_BEGIN                = 0;
	const Time	OMURTAG_T_END                  = 0.3;
	const Time	OMURTAG_T_NETWORK_STEP         = 1e-3;
	const Time	OMURTAG_T_REPORT               = 1e-3;

	const CanvasParameter 
			CANVAS_OMURTAG 
			(
				0.0,			//! t_begin
				OMURTAG_T_END,	//! t_end
				0.0,			//! f_min
				20.0,			//! f_max
				0.0,			//! v_min
				1.0,			//! v_max
				0.0,			//! dense_min
				4.0				//! dense_max
			);

	const RootReportHandler 
		OMURTAG_HANDLER
		(
			"test/omurtag.root",
			TEST_RESULTS_ON_SCREEN,
			true,
			CANVAS_OMURTAG
		);

	const PopulationConnection 
		OMURTAG_WEIGHT
		(
			1,
			OMURTAG_EFFICACY
		);

	const Number	OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS = 10000000;


	// standard test
	const PopulistSpecificParameter
		OMURTAG_SPECIFIC
		(
			OMURTAG_V_MIN,
			OMURTAG_NUMBER_INITIAL_BINS,
			OMURTAG_NUMBER_OF_BINS_TO_ADD,
			OMURTAG_INITIAL_DENSITY,
			OMURTAG_EXPANSION_FACTOR
		);

	const SimulationRunParameter 
		OMURTAG_PARAMETER_RUN
		(
			OMURTAG_HANDLER,
			OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
			OMURTAG_T_BEGIN,
			OMURTAG_T_END,
			OMURTAG_T_REPORT,
			OMURTAG_T_REPORT,
			OMURTAG_T_NETWORK_STEP,
			"test/omurtag.log"
		);

	// fit test

	const FitRateComputation rate_fit;

	const PopulistSpecificParameter
		OMURTAG_FIT
		(
			OMURTAG_V_MIN,
			OMURTAG_NUMBER_INITIAL_BINS,
			OMURTAG_NUMBER_OF_BINS_TO_ADD,
			OMURTAG_INITIAL_DENSITY,
			OMURTAG_EXPANSION_FACTOR
		);

	const RootReportHandler 
		OMURTAG_FIT_HANDLER
		(
			"test/omurtagfit.root",
			TEST_RESULTS_ON_SCREEN,
			true,
			CANVAS_OMURTAG
		);

	const RootReportHandler
		OMURTAG_DOUBLE_REBIN_HANDLER
		(
			"test/omurtag_double_rebin.root",
			TEST_RESULTS_ON_SCREEN,
			false
		);

	// polynomial

	const RootReportHandler 
		OMURTAG_HANDLER_POLYNOMIAL
		(
			"test/omurtag_polynomial.root",
			TEST_RESULTS_ON_SCREEN,
			true,
			CANVAS_OMURTAG
		);

	const PopulistSpecificParameter
		OMURTAG_SPECIFIC_POLYNOMIAL
		(
			OMURTAG_V_MIN,
			OMURTAG_NUMBER_INITIAL_BINS,
			OMURTAG_NUMBER_OF_BINS_TO_ADD,
			OMURTAG_INITIAL_DENSITY,
			OMURTAG_EXPANSION_FACTOR,
			"LIFZeroLeakEquations",
			"PolynomialCirculant",
			"LimitedNonCirculant"
		);

	const SimulationRunParameter 
		OMURTAG_PARAMETER_POLYNOMIAL
		(
			OMURTAG_HANDLER_POLYNOMIAL,
			OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
			OMURTAG_T_BEGIN,
			OMURTAG_T_END,
			OMURTAG_T_REPORT,
			OMURTAG_T_REPORT,
			OMURTAG_T_NETWORK_STEP,
			"test/omurtagpolynomial.log"
		);

	// matrix
	const MatrixNonCirculant non_circulant_matrix;

	const RootReportHandler 
		OMURTAG_HANDLER_MATRIX
		(
			"test/omurtag_polynomial.root",
			TEST_RESULTS_ON_SCREEN,
			true,
			CANVAS_OMURTAG
		);

	const PopulistSpecificParameter
		OMURTAG_SPECIFIC_MATRIX
		(
			OMURTAG_V_MIN,
			OMURTAG_NUMBER_INITIAL_BINS,
			OMURTAG_NUMBER_OF_BINS_TO_ADD,
			OMURTAG_INITIAL_DENSITY,
			OMURTAG_EXPANSION_FACTOR,
			"LIFZeroLeakEquations",
			"CirculantSolver",
			"MatrixNonCirculant"
		);

	const SimulationRunParameter 
		OMURTAG_PARAMETER_MATRIX
		(
			OMURTAG_HANDLER_MATRIX,
			OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
			OMURTAG_T_BEGIN,
			OMURTAG_T_END,
			OMURTAG_T_REPORT,
			OMURTAG_T_REPORT,
			OMURTAG_T_NETWORK_STEP,
			"test/omurtagmatrix.log"
		);
// numerical
	const RootReportHandler 
		OMURTAG_HANDLER_NUMERICAL
		(
			"test/omurtag_polynomial.root",
			TEST_RESULTS_ON_SCREEN,
			true,
			CANVAS_OMURTAG
		);

	const PopulistSpecificParameter
		OMURTAG_SPECIFIC_NUMERICAL
		(
			OMURTAG_V_MIN,
			OMURTAG_NUMBER_INITIAL_BINS,
			OMURTAG_NUMBER_OF_BINS_TO_ADD,
			OMURTAG_INITIAL_DENSITY,
			OMURTAG_EXPANSION_FACTOR,
			"NumericalZeroLeakEquations"
		);

	const SimulationRunParameter 
		OMURTAG_PARAMETER_NUMERICAL
		(
			OMURTAG_HANDLER_NUMERICAL,
			OMURTAG_MAXIMUM_NUMBER_OF_ITERATIONS,
			OMURTAG_T_BEGIN,
			OMURTAG_T_END,
			OMURTAG_T_REPORT,
			OMURTAG_T_REPORT,
			OMURTAG_T_NETWORK_STEP,
			"test/omurtagnumerical.log"
		);

}

#endif // include guard
