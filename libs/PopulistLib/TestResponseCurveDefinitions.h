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
#ifndef _CODE_LIBS_POPULISTLIB_TESTRESPONSECURVEDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TESTRESPONSECURVEDEFINITIONS_INCLUDE_GUARD

#include "IntegralRateComputation.h"
#include "TestInhibitionDefinitions.h"

namespace PopulistLib {


	// at the moment these definitions apply to all responsecurve tests
	const bool INFILE = true;
	const bool ONSCREEN = TEST_RESULTS_ON_SCREEN;

	const InitialDensityParameter
		RESPONSE_CURVE_INITIAL_DENSITY
		(
			0.0,
			0.0
		);

	// Generic simulation parameters
	const Time RESPONSE_CURVE_T_BEGIN   = 0.0;
	const Time RESPONSE_CURVE_T_END     = 0.3;
	const Time RESPONSE_CURVE_T_REPORT  = 1e-3;
	const Time RESPONSE_CURVE_T_NETWORK = 1e-3;

	// Get the neuron parameter values from the ResponseCurve test
	const PopulationParameter 
		PARAMETER_NEURON
		(
			RESPONSE_CURVE_PARAMETER.theta,
			RESPONSE_CURVE_PARAMETER.V_reset,
			0,
			RESPONSE_CURVE_PARAMETER.tau_refractive,
			RESPONSE_CURVE_PARAMETER.tau
		);


	const CanvasParameter 
		RESPONSE_CANVAS 
		(
			0.0,
			1.0,
			0.0,
			25.0,
			0.0,
			20e-3,
			0.0,
			300.0
		);

	const double    RESPONSE_CURVE_EXPANSION_FACTOR	= 1.1;		// maximum expansion before rebinning
	const Number    RESPONSE_CURVE_NADD				= 1;		// add one bin at a time
	const Number	RESPONSE_CURVE_MAX_ITER			= 1000000;	// maximum number of iterations allowed

	const Number    RESPONSE_CURVE_SINGLE_NBINS		= 1100;  // number of bins at start of simulation
	const Number	RESPONSE_CURVE_DOUBLE_NBINS		= 1100;
	const Potential RESPONSE_CURVE_V_MIN			= -0.1*RESPONSE_CURVE_PARAMETER.theta;
	const Density   RESPONSE_CURVE_D_MAX             = 100;

	const string RESPONSE_CURVE_LOG_PATH	("test/response_curve.log");
	const string RESPONSE_CURVE_ROOT_FILE	("test/response_curve.root");

} // end of PopulistLib

#endif // include guard
