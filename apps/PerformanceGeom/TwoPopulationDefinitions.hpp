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
#ifndef _CODE_APPS_PERFORMANCEGEOM_TWOPOPULATIONTEST_INCLUDE_GUARD
#define _CODE_APPS_PERFORMANCEGEOM_TWOPOPULATIONTEST_INCLUDE_GUARD

#include <GeomLib.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

namespace PerformanceGeom {

  const MPILib::Time TWOPOP_T_BEGIN   = 0;
  const MPILib::Time TWOPOP_T_END     = 0.05;
  const MPILib::Time TWOPOP_T_REPORT  = 1e-5;
  const MPILib::Time TWOPOP_T_UPDATE  = 1e-5;
  const MPILib::Time TWOPOP_T_NETWORK = 1e-6;



	// TwoPopulationTest

  const GeomLib::NeuronParameter
		TWOPOP_NET_EXC_PAR
		(
			20e-3, // V_threshold: 20 mV
			0,     // V_reset: 0 mV
			0,     // V_reversal
			0.0,  // tau refractive
			10e-3  // tau membrane; 10 ms
		);

  const GeomLib::NeuronParameter
		TWOPOP_NET_INH_PAR
		(
			20e-3,  // V_threshold; 20 mV
			0,      // V_reset: 0 mV
			0,      // V_reversal
			0.0,   // tau refractive
			3e-3    // tau membrane 3 ms
		);


const GeomLib::InitialDensityParameter
		TWOPOP_INITIAL_DENSITY
		(
			0.0,
			0.0
		);

  const Number TWOPOP_NUMBER_BINS = 5000;
  const Potential TWOPOP_V_MIN    = -1.0*TWOPOP_NET_EXC_PAR._theta;

  const GeomLib::OdeParameter ODE_TWO_EXC(TWOPOP_NUMBER_BINS,TWOPOP_V_MIN,TWOPOP_NET_EXC_PAR,TWOPOP_INITIAL_DENSITY);
  const GeomLib::OdeParameter ODE_TWO_INH(TWOPOP_NUMBER_BINS,TWOPOP_V_MIN,TWOPOP_NET_INH_PAR,TWOPOP_INITIAL_DENSITY);

  const GeomLib::DiffusionParameter EXC_DIFF(0.03, 0.03);
  const GeomLib::DiffusionParameter INH_DIFF(0.03, 0.03);

  const MPILib::Rate RATE_TWOPOPULATION_EXCITATORY_BACKGROUND = 2.0; // Hz

  const double TWOPOPULATION_FRACTION = 0.5;

  const double TWOPOPULATION_C_E = 20000;
  const double TWOPOPULATION_C_I = 2000;

  const MPILib::Efficacy TWOPOPULATION_J_EE = 20e-3/170.0;
  const MPILib::Efficacy TWOPOPULATION_J_IE = 20e-3/70.15;

  const double g = 3.0;

  const MPILib::Efficacy TWOPOPULATION_J_EI = g*TWOPOPULATION_J_EE;
  const MPILib::Efficacy TWOPOPULATION_J_II = g*TWOPOPULATION_J_IE;

  // prevent endless loop
  const Number TWOPOP_NUMBER_INTEGRATION_MAXIMUM = 1000000;

}

#endif // include guard
