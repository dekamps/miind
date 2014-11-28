
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
#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include <GeomLib.hpp>

using namespace GeomLib;


BOOST_AUTO_TEST_CASE(LifNeuralDynamicsConstructorTest) {

	// Test constructor of LifNeuralDynamics and indirectly of AbstractNeuralDynamics
	OrnsteinUhlenbeckParameter par_neuron(20e-3, 0., 0., 0., 10e-3);

	Number n_bins = 5;
	Potential V_min = 0.0;
	InitialDensityParameter par_density(0.0,0.0);

	OdeParameter par_ode(n_bins, V_min, par_neuron, par_density);

	LifNeuralDynamics dyn(par_ode,0.01);
}

BOOST_AUTO_TEST_CASE(LifNeuralDynamicsCurrentComepnesationTest) {
	// Test whether the AbstractNeuralDynamics object provides the correct current compensation object:
	// no current compensation

	OrnsteinUhlenbeckParameter par_neuron(20e-3, 0., 0., 0., 10e-3);

	Number n_bins = 5;
	Potential V_min = 0.0;
	InitialDensityParameter par_density(0.0,0.0);

	OdeParameter par_ode(n_bins, V_min, par_neuron, par_density);

	LifNeuralDynamics dyn(par_ode,0.01);
	CurrentCompensationParameter par_curr = dyn.ParCur();

	BOOST_CHECK(par_curr.NoCurrentCompensation());
}

BOOST_AUTO_TEST_CASE(LifNeuralDynamicsBinLimitsTest){
	//Test whether LifNeuralDynamics generates the expected bin limits


	Potential theta    = 20e-3;
	MPILib::Time tau   = 10e-3;
	OrnsteinUhlenbeckParameter par_neuron(theta, 0., 0., 0., tau);

	Number n_bins = 5;
	Potential V_min = 0.0;
	InitialDensityParameter par_density(0.0,0.0);

	OdeParameter par_ode(n_bins, V_min, par_neuron, par_density);

	double frac = 0.01;
	LifNeuralDynamics dyn(par_ode, frac);

	vector<double> vec_inter = dyn.InterpretationArray();

	BOOST_CHECK( vec_inter.size() == n_bins);


	BOOST_CHECK(vec_inter[0] == 0.0);
	BOOST_CHECK(vec_inter[1] == frac*theta);

	MPILib::Time t_period  = tau*log(1./frac);
	MPILib::Time t_step    = t_period/(n_bins - 1);


	Potential decay_1 = theta*exp(-t_step/tau);

	BOOST_CHECK( decay_1 ==  vec_inter[n_bins-1] );
}
