
// Copyright (c) 2005 - 2014 Marc de Kamps
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
#define BOOST_TEST_DYN_LINK
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include <GeomLib.hpp>

using namespace GeomLib;


BOOST_AUTO_TEST_CASE(OdeParameterTest ) 
{
	// start with OdeParameter
	Number n_plus = 10;

	Time t_mem        =  10e-3;
	Time t_ref        =  0.0;
	Potential V_min   = -0.1;
	Potential V_peak  =  20e-3;
	Potential V_reset =  0.0;
	Potential V_rev   =  0.0;

	NeuronParameter par_neuron(V_peak, V_rev, V_reset, t_ref, t_mem);

	OdeParameter
		par_ode
		(
		 n_plus,
		 V_min,
		 par_neuron,
		 InitialDensityParameter(V_reset,0.0)
		);

	double lambda = 0.01;

	LifNeuralDynamics dyn(par_ode,lambda);
	LeakingOdeSystem syst(dyn);

	// then create an OdeDtParameter and see if we achieve the same effect
	MPILib::Time dt = syst.TStep();

}

