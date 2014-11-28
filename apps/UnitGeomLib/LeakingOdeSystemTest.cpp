
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
#include <GeomLib.hpp>

using namespace GeomLib;
using namespace MPILib;

BOOST_AUTO_TEST_CASE(LeakingOdeSystemCreationTest){

	Number n_plus = 3;

	Time t_mem        =  10e-3;
	Time t_ref        =  0.0;
	Potential V_min   = -0.1;
	Potential V_peak  =  20e-3;
	Potential V_reset =  0.0;
	Potential V_rev   =  0.0;

	PopulationParameter par_neuron(V_peak, V_rev, V_reset, t_ref, t_mem);

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

	pair<Number,Number> bins = syst.BinDistribution();

	BOOST_CHECK(bins.first == 3 && bins.second == 4);
	BOOST_CHECK_CLOSE(syst.TStep(),0.0230259,1e-3);

	double v_0  = syst.InterpretationBuffer()[0];
	double v_1  = syst.InterpretationBuffer()[4];
	double v_2  = syst.InterpretationBuffer()[6];

	BOOST_CHECK_CLOSE(v_0, -0.2, 1e-3);
	BOOST_CHECK      (v_1 == 0.0 );
	BOOST_CHECK_CLOSE(v_2, 0.002, 1e-3);

	BOOST_CHECK( syst.IndexResetBin() == 4);

	BOOST_CHECK(syst.MassBuffer()[0] == 0.0);
	BOOST_CHECK(syst.MassBuffer()[3] == 0.0);
	BOOST_CHECK(syst.MassBuffer()[4] == 1.0);
	BOOST_CHECK(syst.MassBuffer()[5] == 0.0);

}

BOOST_AUTO_TEST_CASE(DensityInitialization)
{
	Number n_plus = 3;

	Time t_mem        =  10e-3;
	Time t_ref        =  0.0;
	Potential V_min   = -0.1;
	Potential V_peak  =  20e-3;
	Potential V_reset =  0.0;
	Potential V_rev   =  0.0;

	PopulationParameter par_neuron(V_peak, V_rev, V_reset, t_ref, t_mem);

	OdeParameter
		par_ode
		(
		 n_plus,
		 V_min,
		 par_neuron,
		 InitialDensityParameter(V_reset,0.0)
		);

	double lambda = 0.01;

	LifNeuralDynamics dyn(par_ode, lambda);
	LeakingOdeSystem syst(dyn);

	pair<Number,Number> number_bins = syst.BinDistribution();

	vector<Potential> array_interpretation(number_bins.first + number_bins.second);
	vector<Potential> array_density(number_bins.first + number_bins.second);

	syst.PrepareReport(&(array_interpretation[0]),&(array_density[0]));
}

BOOST_AUTO_TEST_CASE(BinMapTest)
{
	Number n_plus = 3;

	Time t_mem        =  10e-3;
	Time t_ref        =  0.0;
	Potential V_min   = -0.1;
	Potential V_peak  =  20e-3;
	Potential V_reset =  0.0;
	Potential V_rev   =  0.0;

	PopulationParameter par_neuron(V_peak, V_rev, V_reset, t_ref, t_mem);

	OdeParameter
		par_ode
		(
		 n_plus,
		 V_min,
		 par_neuron,
		 InitialDensityParameter(V_reset,0.0)
		);

	double lambda = 0.01;

	LifNeuralDynamics dyn(par_ode, lambda);
	LeakingOdeSystem syst(dyn);
	pair<Number,Number> number_bins = syst.BinDistribution();

	Number n_bins = number_bins.first + number_bins.second;

	for (Index i = 0; i < n_bins; i++)
		BOOST_CHECK( syst.MapProbabilityToPotentialBin(i) == i );

	syst.Evolve(syst.TStep());


	BOOST_CHECK(syst.MapProbabilityToPotentialBin(0) == 1);
	BOOST_CHECK(syst.MapProbabilityToPotentialBin(1) == 2);
	BOOST_CHECK(syst.MapProbabilityToPotentialBin(2) == 3);
	BOOST_CHECK(syst.MapProbabilityToPotentialBin(3) == 0);
	BOOST_CHECK(syst.MapProbabilityToPotentialBin(4) == 6);
	BOOST_CHECK(syst.MapProbabilityToPotentialBin(5) == 4);
	BOOST_CHECK(syst.MapProbabilityToPotentialBin(6) == 5);

	for (Index i = 0 ; i < n_bins; i++)
		BOOST_CHECK( syst.MapProbabilityToPotentialBin(syst.MapPotentialToProbabilityBin(i)) == i);
}


BOOST_AUTO_TEST_CASE(TestReversalScoop)
{
	Number n_plus = 3;

	Time t_mem        =  10e-3;
	Time t_ref        =  0.0;
	Potential V_min   = -0.1;
	Potential V_peak  =  20e-3;
	Potential V_reset =  0.0;
	Potential V_rev   =  0.0;

	PopulationParameter par_neuron(V_peak, V_rev, V_reset, t_ref, t_mem);

	OdeParameter
		par_ode
		(
		 n_plus,
		 V_min,
		 par_neuron,
		 InitialDensityParameter(V_peak,0.0)
		);

	double lambda = 0.01;

	LifNeuralDynamics dyn(par_ode, lambda);
	LeakingOdeSystem syst(dyn);
	pair<Number,Number> number_bins = syst.BinDistribution();



	BOOST_CHECK( syst.MassBuffer()[number_bins.first + number_bins.second - 1] == 1.0);

	syst.Evolve(syst.TStep()); // t = 1
	syst.Evolve(syst.TStep()); // t = 2
	BOOST_CHECK( syst.MassBuffer()[number_bins.first + number_bins.second - 1] == 1.0);
	syst.Evolve(syst.TStep()); // t = 3

	// after N_+ evolutions, the probability mass should be in the reversal bin,
	// and stay there, i.e. it moves along with the reversal bin from now on.
	BOOST_CHECK( syst.MassBuffer()[number_bins.first + number_bins.second - 1] == 0.0) ;
	BOOST_CHECK( syst.MassBuffer()[syst.FindBin(V_rev)] == 1.0 );

	syst.Evolve(syst.TStep()); // t = 4
	BOOST_CHECK (syst.MassBuffer()[5] == 1.0);

	syst.Evolve(syst.TStep()); // t = 5
	BOOST_CHECK( syst.MassBuffer()[6] == 1.0 );
}

BOOST_AUTO_TEST_CASE(DecayTest){
	Number n_plus = 3;

	Time t_mem        =  10e-3;
	Time t_ref        =  0.0;
	Potential V_min   = -0.1;
	Potential V_peak  =  20e-3;
	Potential V_reset =  0.0;
	Potential V_rev   =  0.0;

	PopulationParameter par_neuron(V_peak, V_rev, V_reset, t_ref, t_mem);

	OdeParameter
		par_ode
		(
		 n_plus,
		 V_min,
		 par_neuron,
		 InitialDensityParameter(V_min,0.0)
		);

	double lambda = 0.01;

	LifNeuralDynamics dyn(par_ode, lambda);
	LeakingOdeSystem syst(dyn);
	pair<Number,Number> number_bins = syst.BinDistribution();

	Number n_bins = number_bins.first + number_bins.second;

	vector<double> array_interpretation(n_bins);
	vector<double> array_mass(n_bins);

	syst.PrepareReport(&array_interpretation[0],&array_mass[0]);
	BOOST_CHECK (array_mass[0]  > 0.0);
	for (Index i = 1; i < 5; i++)
		BOOST_CHECK (array_mass[i] == 0);

	syst.Evolve(syst.TStep());
	syst.PrepareReport(&array_interpretation[0],&array_mass[0]);


	BOOST_CHECK(array_mass[1] > 0.0);
	for (Index i = 0; i < 5; i++)
		if (i != 1)
			BOOST_CHECK(array_mass[i] == 0.0);

	syst.Evolve(syst.TStep());
	syst.PrepareReport(&array_interpretation[0],&array_mass[0]);
	syst.Evolve(syst.TStep());
	syst.PrepareReport(&array_interpretation[0],&array_mass[0]);
	syst.Evolve(syst.TStep());
	syst.PrepareReport(&array_interpretation[0],&array_mass[0]);
	syst.Evolve(syst.TStep());
	syst.PrepareReport(&array_interpretation[0],&array_mass[0]);


	// Although 5 evolutions have now taken place, the mass remains
	// in the reset bin (i=3). Above it was decaying towards it.
	BOOST_CHECK( array_mass[3] > 0.0 );
	for (Index i = 0; i < 5; i++ )
		if (i != 3)
			BOOST_CHECK( array_mass[i] == 0.0 );
}

