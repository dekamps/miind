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
#ifndef _CODE_APPS_LARGENETWORK_INCLUDE_GUARD
#define _CODE_APPS_LARGENETWORK_INCLUDE_GUARD


#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/parameters/PopulistParameter.hpp>
#include <MPILib/include/populist/parameters/InitialDensityParameter.hpp>

namespace largeNetwork {
	
	const MPILib::populist::parameters::OrnsteinUhlenbeckParameter
		TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER 
		(
			20e-3, // V_threshold: 20 mV
			0,     // V_reset: 0 mV
			0,     // V_reversal
			2e-3,  // tau refractive
			10e-3  // tau membrane; 10 ms
		);

	const MPILib::populist::parameters::OrnsteinUhlenbeckParameter
		TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER
		(
			20e-3,  // V_threshold; 20 mV
			0,      // V_reset: 0 mV
			0,      // V_reversal
			2e-3,   // tau refractive
			3e-3    // tau membrane 3 ms
		);

	const MPILib::Rate RATE_TWOPOPULATION_EXCITATORY_BACKGROUND = 2.0; // Hz

	const double TWOPOPULATION_FRACTION = 0.5;

	const double TWOPOPULATION_C_E = 20000;
	const double TWOPOPULATION_C_I = 2000;

	const MPILib::Efficacy TWOPOPULATION_J_EE = 20e-3/170.0;
	const MPILib::Efficacy TWOPOPULATION_J_IE = 20e-3/70.15;

	const double g = 3.0;

	const double T_DELAY = 0.0;

	const MPILib::Efficacy TWOPOPULATION_J_EI = g*TWOPOPULATION_J_EE;
	const MPILib::Efficacy TWOPOPULATION_J_II = g*TWOPOPULATION_J_IE;

	const MPILib::populist::parameters::InitialDensityParameter
		TWOPOP_INITIAL_DENSITY
		(
			0.0,
			0.0
		);

	const MPILib::Number TWOPOP_NUMBER_OF_INITIAL_BINS		= 1100; //crank this up if you want to have higher CPU load
	const MPILib::Number TWOPOP_NUMBER_OF_BINS_TO_ADD		= 1;
	const MPILib::Number TWOPOP_MAXIMUM_NUMBER_OF_ITERATIONS	= 1000000;

	const double TWOPOP_EXPANSION_FACTOR = 1.1;

	const MPILib::Potential TWOPOP_V_MIN  = -1.0*TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER._theta;

	const MPILib::populist::parameters::PopulistSpecificParameter
		TWOPOP_SPECIFIC
		(
			TWOPOP_V_MIN,
			TWOPOP_NUMBER_OF_INITIAL_BINS,
			TWOPOP_NUMBER_OF_BINS_TO_ADD,
			TWOPOP_INITIAL_DENSITY,
			TWOPOP_EXPANSION_FACTOR,
			"NumericalZeroLeakEquations"
		);

    const MPILib::populist::parameters::PopulistParameter
		TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER_POP
		(
			TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER,
			TWOPOP_SPECIFIC
		);

	const MPILib::populist::parameters::PopulistParameter
		TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER_POP
		(
			TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER,
			TWOPOP_SPECIFIC
		);

	const double BURST_FACTOR = 0.05;
}

#endif // include guard
