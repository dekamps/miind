// Copyright (c) 2005 - 2010 Marc de Kamps
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
#ifndef _CODE_LIBS_POPULISTLIB_TESTINHIBITIONDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TESTINHIBITIONDEFINITIONS_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "InterpolationRebinner.h"
#include "IntegralRateComputation.h"
#include "TestBinCalculationDefinitions.h"

using DynamicLib::RootReportHandler;

namespace PopulistLib {

	// Inhibition test

	const PopulationParameter 
		INHIBITION_PARAMETER
		(
			1,                     // theta
			0,                     // V_reset
			0,                     // V_reversal
			0,                     // no refractive period
			TIME_MEMBRANE_INFINITE // 50 ms or 20 s^-1
		);
	const InitialDensityParameter
		INHIBITION_INITIAL_DENSITY
		(
			1.0,
			0.0
		);
	const Time INHIBITION_T_BEGIN   = 0.0;
	const Time INHIBITION_T_END     = 0.5;
	const Time INHIBITION_T_REPORT  = 0.01;
	const Time INHIBITION_T_NETWORK = 0.01;

	const double    INHIBITION_EXPANSION_FACTOR       = 1.2;  // maximum expansion before rebinning
	const Number    INHIBITION_NUMBER_OF_BINS_TO_ADD  = 1;    // add one bin at a time


	const Number    INHIBITION_NUMBER_INITIAL_BINS = 330;  // number of bins at start of simulation
	const Potential INHIBITION_V_MIN               = -0.1;

	const Rate     INHIBITION_RATE     = 10.0;
	const Efficacy INHIBITION_EFFICACY = -0.03;

	const CanvasParameter 
		INHIBITION_CANVAS 
		(
			0.0,	//! t_min
			10.0,	//! t_max
			0.0,	//! f_min
			1.0,    //! f_max
			0.0,	//! state_min
			1.0,	//! state_max
			0.0,	//! dense_min
			10.0	//! dense_max
		);

	const RootReportHandler 
		INHIBITION_HANDLER
		(
			"test/inhibition.root",
			false,
			TEST_RESULTS_ON_SCREEN,
			INHIBITION_CANVAS
		);

	const Number	INHIBITION_MAXIMUM_NUMBER_OF_ITERATIONS = 10000000;


	const PopulistSpecificParameter
		INHIBITION_SPECIFIC
		(
			INHIBITION_V_MIN,
			INHIBITION_NUMBER_INITIAL_BINS,
			INHIBITION_NUMBER_OF_BINS_TO_ADD,
			INHIBITION_INITIAL_DENSITY,
			INHIBITION_EXPANSION_FACTOR
		);

	const SimulationRunParameter 
		INHIBITION_PARAMETER_RUN
		(
			INHIBITION_HANDLER,
			INHIBITION_MAXIMUM_NUMBER_OF_ITERATIONS,
			INHIBITION_T_BEGIN,
			INHIBITION_T_END,
			INHIBITION_T_REPORT,
			INHIBITION_T_REPORT,
			INHIBITION_T_NETWORK,
			"test/inhibition.log"//,
//			&INHIBITION_SPECIFIC
		);
}

#endif // include guard
