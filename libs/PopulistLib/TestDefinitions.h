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
#ifndef _CODE_LIBS_POPULISTLIB_TESTDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TESTDEFINITIONS_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "../UtilLib/UtilLib.h"
#include "BasicDefinitions.h"
#include "FitRateComputation.h"
#include "IntegralRateComputation.h"
#include "LimitedNonCirculant.h"
#include "OrnsteinUhlenbeckParameter.h"
#include "PolynomialCirculant.h"
#include "ResponseParameterBrunel.h"

using DynamicLib::AlgorithmGrid;
using DynamicLib::InactiveReportHandler;
using DynamicLib::SimulationRunParameter;
using UtilLib::Number;


namespace PopulistLib {
	// Macro ProcessVData relies on specific values for these constants
	// In general, the values below need to be taken over there

	//! This variable controls wether the test results are visualized in a ROOT screen. In new releases this variable should be false.
	const bool TEST_RESULTS_ON_SCREEN = false;

	const int VDATA_NCIRC  = 51;
	const int VDATA_NSTEPS = 100;
	const Time VDATA_TMIN  = 1e-9;
	const Time VDATA_TMAX  = 10;
	const Time VDATA_JMAX  = 10;

	const Number NUMBER_OF_RESERVED_NODES = 1000;

	const double SIGMA  = 2.0e-3F;  

	const ResponseParameterBrunel 
		RESPONSE_CURVE_PARAMETER = 
		{
			0,		// mu
			SIGMA,	// sigma
			20e-3F,	// theta
			10e-3F,	// V_reset
			0,		// V_reversal
			0.004F,	// tau ref
			0.020F	// tau exc
		};    

	const double FRACT = 0.1F;
	const double N_EXC   = 16000;
	const double N_INH   = 4000;
	const double J_EE    = 0.05e-3F;
	const double J_EI    = 3*J_EE;

	// Default values reproduce the response curve from
	// figure 3 of Brunel, N. (2000), Network: Comput. Neural Syst. 11 (2000) 261-280.


	// Given the parameters above, these functions produce rates that in turn produce
	// mu and sigma
	// Here are 6 points of that curve

	const Number NUMBER_RESPONSE_CURVE_POINTS = 8;

	const float MU[NUMBER_RESPONSE_CURVE_POINTS] = 
			{
				15e-3F, 
				16e-3F, 
				17e-3F, 
				18e-3F, 
				19e-3F, 
				20e-3F,
				21e-3F,
				22e-3F
			};

	const float RESPONSE_REFRACTIVE[NUMBER_RESPONSE_CURVE_POINTS] = 
			{
				0.12199584F,
				0.84888909F,
				3.25051013F,
				7.55144F,
				12.7002F,
				17.84175F
			};

	const float RESPONSE_NON_REFRACTIVE[NUMBER_RESPONSE_CURVE_POINTS] =
			{
				0.122055401F,
				0.851781362F,
				3.293330142F,
				7.786641422F,
				13.37991014F,
				19.21291834F
			};


	inline Rate test_rate_e
	(
		Potential f_mu, 
		Potential f_sigma = SIGMA 
	)
	{
		Rate nu_e = (f_sigma*f_sigma + J_EI*f_mu)/(N_EXC*J_EE*(J_EE+J_EI)*RESPONSE_CURVE_PARAMETER.tau);
		return nu_e;
	}

	inline Rate test_rate_i
	(
		Potential f_mu, 
		Potential f_sigma = SIGMA 
	)
	{ 
		Rate nu_i = (f_sigma*f_sigma - J_EE*f_mu)/(N_INH*J_EI*(J_EE+J_EI)*RESPONSE_CURVE_PARAMETER.tau);
		return nu_i;
	}

	const InactiveReportHandler no_handling;

	const SimulationRunParameter 
		TEST_ORNSTEIN_RUN_PARAMETER
		(
			no_handling,
			100000,
			0,
			1,
			1,
			1,
			1e-4,
			"test/ornsteintest.log"
		);

	// TwoPopulationTest
	
	const OrnsteinUhlenbeckParameter 
		TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER 
		(
			20e-3, // V_threshold: 20 mV
			0,     // V_reset: 0 mV
			0,     // V_reversal
			2e-3,  // tau refractive
			10e-3  // tau membrane; 10 ms
		);

	const OrnsteinUhlenbeckParameter 
		TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER
		(
			20e-3,  // V_threshold; 20 mV
			0,      // V_reset: 0 mV
			0,      // V_reversal
			2e-3,   // tau refractive
			3e-3    // tau membrane 3 ms
		);

	const Rate RATE_TWOPOPULATION_EXCITATORY_BACKGROUND = 2.0; // Hz

	const double TWOPOPULATION_FRACTION = 0.5;

	const double TWOPOPULATION_C_E = 20000;
	const double TWOPOPULATION_C_I = 2000;

	const Efficacy TWOPOPULATION_J_EE = 20e-3/170.0;
	const Efficacy TWOPOPULATION_J_IE = 20e-3/70.15;

	const double g = 3.0;

	const Efficacy TWOPOPULATION_J_EI = g*TWOPOPULATION_J_EE;
	const Efficacy TWOPOPULATION_J_II = g*TWOPOPULATION_J_IE;

	const Time TWOPOPULATION_TIME_BEGIN   = 0;    // 0 sec
	const Time TWOPOPULATION_TIME_END     = 0.05;    // 1 sec
	const Time TWOPOPULATION_TIME_REPORT  = 1e-3; // 10 ms
	const Time TWOPOPULATION_TIME_UPDATE  = 1e-2; // 100 ms
	const Time TWOPOPULATION_TIME_NETWORK = 1e-6; // 0.1 ms

	// prevent endless loop
	const Number NUMBER_INTEGRATION_MAXIMUM = 1000000;

	const AlgorithmGrid TWOPOPULATION_GRID(1);

	const string STRING_TEST_DIR
			(
				"test/"
			);

	const string STR_TWOPOPULATION_DELAY
			(
				"twopopulation.root"
			);

	// Delay inhibition test


	const Efficacy DELAY_J_EE = 20e-3/193.3;
	const Efficacy DELAY_J_IE = 20e-3/120;

	const OrnsteinUhlenbeckParameter 
		DELAY_EXCITATORY_PARAMETER 
		(
			20e-3, // V_threshold: 20 mV
			0,     // V_reset: 0 mV
			0,     // V_reversal
			2e-3,   // tau refractive
			10e-3 // tau membrane; 10 ms
		);

	const OrnsteinUhlenbeckParameter 
		DELAY_INHIBITORY_PARAMETER
		(
			20e-3, // V_threshold; 20 mV
			0,     // V_reset: 0 mV
			0,     // V_reversal
			2e-3,  // tau refractive
			5e-3   // tau membrane 5 ms
		);

	const double DELAY_C_E = 20000;
	const double DELAY_C_I = 2000;


	const double g_DELAY = 5.0;

	const Efficacy DELAY_J_EI = g_DELAY*DELAY_J_EE;
	const Efficacy DELAY_J_II = g_DELAY*DELAY_J_IE;


	const double PULSE_REST = 3.0;
	const double PULSE_HIGH = 20.0;

	const DynamicLib::Time TIME_PULSE      = 0.40;
	const DynamicLib::Time TIME_PULSE_LAST = 0.10;


	inline DynamicLib::Rate Pulse(DynamicLib::Time t) {
		if ( t < TIME_PULSE )
			return PULSE_REST;
		else
			if ( t < TIME_PULSE + TIME_PULSE_LAST )
				return PULSE_HIGH;
			else
				return PULSE_REST;
	}

	const Rate DELAY_RATE_CORTICAL_BG = 3;

	const double DELAY_X     = 0.5;
	const double DELAY_X_DA  = 0.02;
	const double DELAY_X_SUP = 0.05;
	const double DELAY_X_DIS = 0.01;
	const double DELAY_X_C   = 0.03;
	const double DELAY_X_DAC = 0.10;

	const double DELAY_GAMMA_DA     = 1.25;
	const double DELAY_GAMMA_I_DIS  = 1.4; 
	const double DELAY_GAMMA_C      = 2.0;
	const double DELAY_GAMMA_DIS    = 1.1;
	const double DELAY_GAMMA_SUP    = 7;
	const double DELAY_GAMMA_DA_DIS = 2.0;

	const string 
		STR_TEST_DIRECTORY
		(
			"test/"
		);

	const string 
		STR_DELAY_DISINHIBITION_NAME
		(
			"delay_disinhibition.root"
		);

	const FitRateComputation      FIT_RATE_COMPUTATION;

	const IntegralRateComputation INTEGRAL_RATE_COMPUTATION;

	// polynomial
	const PolynomialCirculant POLYNOMIAL_CIRCULANT;

	const LimitedNonCirculant LIMITED_NON_CIRCULANT;

} // end of PopulistLib

#endif // include guard

