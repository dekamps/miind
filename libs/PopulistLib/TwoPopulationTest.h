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
#ifndef _CODE_LIBS_POPULISTLIB_TWOPOPULATIONTEST_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TWOPOPULATIONTEST_INCLUDE_GUARD

#include "PopulistParameter.h"
#include "TestDefinitions.h"
#include "TestOmurtagDefinitions.h"
#include "TestResponseCurveDefinitions.h"

namespace PopulistLib {

	const Time TWOPOP_T_BEGIN   = 0;
	const Time TWOPOP_T_END     = 0.05;
	const Time TWOPOP_T_REPORT  = 1e-5;
	const Time TWOPOP_T_NETWORK = 1e-5;

	const InitialDensityParameter
		TWOPOP_INITIAL_DENSITY
		(
			0.0,
			0.0
		);

	const Number TWOPOP_NUMBER_OF_INITIAL_BINS		= 550;
	const Number TWOPOP_NUMBER_OF_BINS_TO_ADD		= 1;
	const Number TWOPOP_MAXIMUM_NUMBER_OF_ITERATIONS	= 1000000;

	const double TWOPOP_EXPANSION_FACTOR = 1.1;

	const Potential TWOPOP_V_MIN  = -1.0*RESPONSE_CURVE_PARAMETER.theta;

	const PopulistSpecificParameter
		TWOPOP_SPECIFIC
		(
			TWOPOP_V_MIN,
			TWOPOP_NUMBER_OF_INITIAL_BINS,
			TWOPOP_NUMBER_OF_BINS_TO_ADD,
			TWOPOP_INITIAL_DENSITY,
			TWOPOP_EXPANSION_FACTOR,
			"NumericalZeroLeakEquations"
		);

    const PopulistParameter
		TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER_POP
		(
			TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER,
			TWOPOP_SPECIFIC
		);

	const PopulistParameter
		TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER_POP
		(
			TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER,
			TWOPOP_SPECIFIC
		);

	const CanvasParameter 
		CANVAS_TWOPOP 
		(
			0.0,
			0.05,
			0.0,
			10.0,
			0.0,
			20e-3,
			0.0,
			200.0
		);

	const RootReportHandler
		TWOPOP_HANDLER
		(
			"test/twopoptest.root",
			TEST_RESULTS_ON_SCREEN,   // on display
			true,	 // in file
			CANVAS_TWOPOP
		);

	const SimulationRunParameter 
		TWOPOP_PARAMETER
		(
			TWOPOP_HANDLER,
			TWOPOP_MAXIMUM_NUMBER_OF_ITERATIONS,
			TWOPOP_T_BEGIN,
			TWOPOP_T_END,
			TWOPOP_T_REPORT,
			TWOPOP_T_REPORT,
			TWOPOP_T_NETWORK,
			"test/twopoptest.log"
		);
}

#endif // include guard
