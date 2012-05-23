// Copyright (c) 2005 - 2009 Marc de Kamps
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
#ifndef _CODE_LIBS_POPULISTLIB_TESTZEROLEAKGAUSSDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TESTZEROLEAKGAUSSDEFINITIONS_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "TestInhibitionDefinitions.h"

using DynamicLib::RootReportHandler;

namespace PopulistLib {

	// ZeroLeakGauss test
	const PopulationParameter 
		ZEROLEAKGAUSS_PARAMETER
		(
			1,                     // theta
			0,                     // V_reset
			0,                     // V_reversal
			0,                     // no refractive period
			TIME_MEMBRANE_INFINITE // 50 ms or 20 s^-1
		);
	const InitialDensityParameter
		ZEROLEAKGAUSS_INITIAL_DENSITY
		(
			0.0,
			0.0
		);
	const Time ZEROLEAKGAUSS_T_BEGIN   = 0.0;
	const Time ZEROLEAKGAUSS_T_END     = 10.0;
	const Time ZEROLEAKGAUSS_T_REPORT  = 0.1;
	const Time ZEROLEAKGAUSS_T_NETWORK = 0.1;

	const double    ZEROLEAKGAUSS_EXPANSION_FACTOR       = 1.0;  // maximum expansion before rebinning
	const Number    ZEROLEAKGAUSS_NUMBER_OF_BINS_TO_ADD  = 1;    // add one bin at a time

	const Number    ZEROLEAKGAUSS_NUMBER_INITIAL_BINS = 1000;  // number of bins at start of simulation
	const Potential ZEROLEAKGAUSS_V_MIN                  = -3.0;

	// mu and sigma defined per unit time !
	const Rate     ZEROLEAKGAUSS_RATE     = 1.0/TIME_MEMBRANE_INFINITE;
	const Efficacy ZEROLEAKGAUSS_EFFICACY = 0.03;

	const CanvasParameter 
		ZEROLEAKGAUSS_CANVAS  
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
		ZEROLEAKGAUSS_HANDLER
		(
			"test/zeroleakgauss.root",
			false,
			true,
			ZEROLEAKGAUSS_CANVAS
		);

	const Number	ZEROLEAKGAUSS_MAXIMUM_NUMBER_OF_ITERATIONS = 10000000;

	const PopulistSpecificParameter
		ZEROLEAKGAUSS_SPECIFIC
		(
			ZEROLEAKGAUSS_V_MIN,
			ZEROLEAKGAUSS_NUMBER_INITIAL_BINS,
			ZEROLEAKGAUSS_NUMBER_OF_BINS_TO_ADD,
			ZEROLEAKGAUSS_INITIAL_DENSITY,
			ZEROLEAKGAUSS_EXPANSION_FACTOR
		);

	const SimulationRunParameter 
		ZEROLEAKGAUSS_PARAMETER_RUN
		(
			ZEROLEAKGAUSS_HANDLER,
			ZEROLEAKGAUSS_MAXIMUM_NUMBER_OF_ITERATIONS,
			ZEROLEAKGAUSS_T_BEGIN,
			ZEROLEAKGAUSS_T_END,
			ZEROLEAKGAUSS_T_REPORT,
			ZEROLEAKGAUSS_T_REPORT,
			ZEROLEAKGAUSS_T_NETWORK,
			"test/zeroleakgauss.log"
		);
}

#endif // include guard
