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
#ifndef _CODE_LIBS_POPULISTLIB_TRANSFERTESTDEFINITION_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TRANSFERTESTDEFINITION_INCLUDE_GUARD

#include "TestBinCalculationDefinitions.h"
#include "TestZeroLeakDefinitions.h"

namespace PopulistLib {

	// Transfer 

	const Number    TRANSFER_NUMBER_INITIAL_BINS = 10;
	const Potential TRANSFER_V_MIN = 0;

	const PopulationParameter 
		TRANSFER_PARAMETER
		(
			1,   // theta
			0,   // V_reset
			0,   // V_reversal
			0,   // no refractive parameter
			TIME_MEMBRANE_INFINITE
		);

	const Time	TRANSFER_T_BEGIN                = 0;
	const Time	TRANSFER_T_END                  = 100;
	const Time	TRANSFER_T_NETWORK_STEP         = 1;
	const Time	TRANSFER_T_REPORT               = 1;

	const double	TRANSFER_EXPANSION_FACTOR         = 1;     // no rescaling necessary
	const Number	TRANSFER_NUMBER_OF_BINS_TO_ADD    = 0;     // no bins to add

	const string 
		STR_TRANSFER_ROOT
		(
			"test/transfer.root",
			false,
			true
		);	

	const Number TRANSFER_MAXIMUM_NUMBER_OF_ITERATIONS = 100000;

	const Rate      TRANSFER_EFFICACY             =  2/(static_cast<double>(TRANSFER_NUMBER_INITIAL_BINS) - 1);   // Hz
	const Efficacy  TRANSFER_RATE                 = 1;                                                            // potential

	const InitialDensityParameter 
		TRANSFER_INITIAL_DENSITY
		(
			0.0,
			0.0
		);

	const PopulistSpecificParameter
		TRANSFER_SPECIFIC
		(
			TRANSFER_V_MIN,
			TRANSFER_NUMBER_INITIAL_BINS,
			TRANSFER_NUMBER_OF_BINS_TO_ADD,
			TRANSFER_INITIAL_DENSITY,
			TRANSFER_EXPANSION_FACTOR,
			"LIFZeroLeakEquations",
			"CirculantSolver",
			"NonCirculantSolver",
			&dont_rebin
		);

	const SimulationRunParameter 
		TRANSFER_PARAMETER_RUN
		(
			handler,
			TRANSFER_MAXIMUM_NUMBER_OF_ITERATIONS,
			TRANSFER_T_BEGIN,
			TRANSFER_T_END,
			TRANSFER_T_REPORT,
			TRANSFER_T_REPORT,
			TRANSFER_T_NETWORK_STEP,
			"test/transfer.log"
		);

} // end of PopulistLib

#endif // include guard
