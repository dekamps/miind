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
#ifndef _CODE_LIBS_POPULIST_TESTBINCALCULATIONDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_POPULIST_TESTBINCALCULATIONDEFINITIONS_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "IntegralRateComputation.h"
#include "PopulistSpecificParameter.h"
#include "SinglePeakRebinner.h"

using namespace DynamicLib;

namespace PopulistLib {

	const InactiveReportHandler handler;

	// BINCALCULATION

	const PopulationParameter 
		BINCALCULATION_PARAMETER
		(
			1,     // theta
			0,     // V_reset
			0,     // V_reversal
			0,     // no refractive period
			50e-3  // 50 ms or 20 s^-1
		);

	const InitialDensityParameter 
		BINCALCULATION_INITIAL_DENSITY
		(
			1,
			0
		);

	const Rate       BINCALCULATION_RATE                   = 0  ;		// Hz							
	const Efficacy   BINCALCULATION_EFFICACY               = 0;			// rescaled potential

	const double     BINCALCULATION_EXPANSION_FACTOR       = 1000;		// maximum expansion before rebinning
	const Number     BINCALCULATION_NUMBER_OF_BINS_TO_ADD  = 1;			// add one bin at a time
	const Number     BINCALCULATION_NUMBER_INITIAL_BINS    = 200;		// number of bins at start of simulation
	const Potential  BINCALCULATION_V_MIN				   = -0.05;		// rescaled minimal potential for grid

	const string STR_BINCALCULATION_ROOT
			(
				"test/bincalculation.root",
				false,
				true
			);



	const Time      BINCALCULATION_T_BEGIN                = 0;
	const Time      BINCALCULATION_T_END                  = 0.1;
	const Time      BINCALCULATION_T_NETWORK_STEP         = 1e-3;
	const Time      BINCALCULATION_T_REPORT               = 1e-2;

	const Number    BINCALCULATION_MAXIMUM_NUMBER_OF_ITERATIONS = 1000000;
	
	const SinglePeakRebinner      rebin_speak;
	const IntegralRateComputation rate_integral;

	const PopulistSpecificParameter
		BINCALCULATION_SPECIFIC
		(
			BINCALCULATION_V_MIN,
			BINCALCULATION_NUMBER_INITIAL_BINS,
			BINCALCULATION_NUMBER_OF_BINS_TO_ADD,
			BINCALCULATION_INITIAL_DENSITY,
			BINCALCULATION_EXPANSION_FACTOR,
			"LIFZeroLeakEquations",
			"CirculantSolver",
			"NonCirculantSolver",
			&rebin_speak,
			&rate_integral
		);

	const SimulationRunParameter 
		BINCALCULATION_PARAMETER_RUN
		(
			handler,
			BINCALCULATION_MAXIMUM_NUMBER_OF_ITERATIONS,
			BINCALCULATION_T_BEGIN,
			BINCALCULATION_T_END,
			BINCALCULATION_T_REPORT,
			BINCALCULATION_T_REPORT,
			BINCALCULATION_T_NETWORK_STEP,
			"test/bincalculation.log"
		);


}

#endif // include guard
