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
#ifndef _CODE_LIBS_POPULISTLIB_TESTZEROLEAKDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TESTZEROLEAKDEFINITIONS_INCLUDE_GUARD

#include "NoopRebinner.h"

namespace PopulistLib {


	const Rate      ZEROLEAK_RATE                   = 100;   // Hz
	const Efficacy  ZEROLEAK_EFFICACY				= 0.01;   // potential
	const double    ZEROLEAK_EXPANSION_FACTOR		= 1;     // no rescaling necessary
	const Number    ZEROLEAK_NUMBER_OF_BINS_TO_ADD	= 0;     // no bins to add


	const Number    ZEROLEAK_NUMBER_INITIAL_BINS	= 101;   // number of bins during simulation
	const Potential ZEROLEAK_V_MIN					= 0;     // No negative extension

	const PopulationParameter 
		ZEROLEAK_PARAMETER
		(
			1,   // theta
			0,   // V_reset
			0,   // V_reversal
			0,   // no refractive parameter
			TIME_MEMBRANE_INFINITE
		);

	const InitialDensityParameter 
		ZEROLEAK_INITIAL_DENSITY
		(
			0,
			0
		);


	const PopulationConnection 
		ZEROLEAK_WEIGHT
		(
			1,
			ZEROLEAK_EFFICACY
		);



	const string STR_ZEROLEAK_ROOT
			(
				"test/zeroleak.root",
				false,
				true
			);


	const Time	ZEROLEAK_T_BEGIN                = 0;
	const Time	ZEROLEAK_T_END                  = 1;
	const Time	ZEROLEAK_T_NETWORK_STEP         = 1e-3;
	const Time	ZEROLEAK_T_REPORT               = 1e-2;

	const Number      ZEROLEAK_MAXIMUM_NUMBER_OF_ITERATIONS = 100000;

	const NoopRebinner dont_rebin;

	const PopulistSpecificParameter
		ZEROLEAK_SPECIFIC
		(
			ZEROLEAK_V_MIN,
			ZEROLEAK_NUMBER_INITIAL_BINS,
			ZEROLEAK_NUMBER_OF_BINS_TO_ADD,
			ZEROLEAK_INITIAL_DENSITY,
			ZEROLEAK_EXPANSION_FACTOR,
			"LIFZeroLeakEquations",
			"CirculantSolver",
			"NonCirculantSolver",
			&dont_rebin
		);

	const SimulationRunParameter 
		ZEROLEAK_PARAMETER_RUN
		(
			handler,
			ZEROLEAK_MAXIMUM_NUMBER_OF_ITERATIONS,
			ZEROLEAK_T_BEGIN,
			ZEROLEAK_T_END,
			ZEROLEAK_T_REPORT,
			ZEROLEAK_T_REPORT,
			ZEROLEAK_T_NETWORK_STEP,
			"test/zeroleak.log"
		);
}

#endif // include guard
