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
#ifndef _CODE_LIBS_POPULISTLIB_TESTOUDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TESTOUDEFINITIONS_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "TestInhibitionDefinitions.h"

using DynamicLib::RootReportHandler;

namespace PopulistLib {

	// OUTest

	const PopulationParameter 
		OUTEST_PARAMETER
		(
			10,					// theta
			0,					// V_reset
			0,					// V_reversal
			0,					// no refractive period
			50e-3				// 50 ms or 20 s^-1
		);
	const InitialDensityParameter
		OUTEST_INITIAL_DENSITY
		(
			0.0,
			0.0
		);
	const Time OUTEST_T_BEGIN   = 0.0;
	const Time OUTEST_T_END     = 5e-3;
	const Time OUTEST_T_REPORT  = 1e-5;
	const Time OUTEST_T_NETWORK = 1e-5;

	const double    OUTEST_EXPANSION_FACTOR       = 10.0;  // maximum expansion before rebinning
	const Number    OUTEST_NUMBER_OF_BINS_TO_ADD  = 1;     // add one bin at a time

	const Number    OUTEST_NUMBER_INITIAL_BINS = 1000;     // number of bins at start of simulation
	const Potential OUTEST_V_MIN               = -4.0;

	// mu and sigma defined per unit time !
	const Rate     OUTEST_RATE     = 1000.0;
	const Efficacy OUTEST_EFFICACY = 0.1;

	const CanvasParameter 
		OU_CANVAS 
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
		OUTEST_HANDLER
		(
			"test/OUtest.root",
			false,
			true,
			OU_CANVAS
	);

	const Number	OUTEST_MAXIMUM_NUMBER_OF_ITERATIONS = 10000000;

	const PopulistSpecificParameter
		OUTEST_SPECIFIC
		(
			OUTEST_V_MIN,
			OUTEST_NUMBER_INITIAL_BINS,
			OUTEST_NUMBER_OF_BINS_TO_ADD,
			OUTEST_INITIAL_DENSITY,
			OUTEST_EXPANSION_FACTOR
		);

	const SimulationRunParameter 
		OUTEST_PARAMETER_RUN
		(
			OUTEST_HANDLER,
			OUTEST_MAXIMUM_NUMBER_OF_ITERATIONS,
			OUTEST_T_BEGIN,
			OUTEST_T_END,
			OUTEST_T_REPORT,
			OUTEST_T_REPORT,
			OUTEST_T_NETWORK,
			"test/outest.log"
		);
}

#endif // include guard
