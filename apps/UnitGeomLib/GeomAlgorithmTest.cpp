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
#include <boost/shared_ptr.hpp>
#include <GeomLib.hpp>
//#include "../UnitDynamicLib/SinglePopulationNetworkFixtureCode.h"
//#include "AlgorithmInputSimulationFixture.h"

using namespace GeomLib;
using namespace MPILib;

//BOOST_FIXTURE_TEST_SUITE( s, SinglePopulationNetworkFixture<OrnsteinUhlenbeckConnection> )

/*
//____________________________________________________________________________//

BOOST_AUTO_TEST_CASE( QIF_Algorithm_Create_And_Copy )
{

	BOOST_REQUIRE( true );
}

//____________________________________________________________________________//

BOOST_AUTO_TEST_CASE(InitializeAlgorithm)
{
	Potential gamma		=  0.5;  
	Potential V_reset	=  0.0;  
	Potential V_peak	= 10.0;
	Potential V_min		= -5.0;
	Time t_mem			= 10e-3;
	Time t_ref			= 0.0;

	QifParameter
		par_qif
		(	
			gamma
		);

	InitialDensityParameter par_dense(0.0,1e10);

	PopulationParameter par_neuron(V_peak, V_reset, V_reset, t_ref, t_mem);
	Number n_bins = 400;

	OdeParameter 
		par_ode
		(
			n_bins,
			V_min,
			par_neuron,
         	par_dense
		);

	SpikingQifNeuralDynamics dyn(par_ode, par_qif);
	QifOdeSystem sys(dyn);

	GeomAlgorithm<MPILib::populist::OrnsteinUhlenbeckConnection> alg(sys);
}
/*
BOOST_AUTO_TEST_CASE(SingleInputTest)
{

	Potential V_min		= -10.0;
	Potential V_peak	=  10.0;
	Potential V_reset	= -10.0;
	Potential gamma		=  0.5;
	Time t_mem			=  10e-3;
	Time  t_ref			= 0.0;

	QifParameter
		par_qif
		(
			gamma
		);

	Number n_bins = 200; // 100 bins not enough; triggers sigma correction

	PopulationParameter par_neuron(V_peak, V_reset, V_reset, t_ref, t_mem);

	InitialDensityParameter par_dense(0.0,1e10);
	OdeParameter 
		par_ode
		(
			n_bins,
			V_min,
			par_neuron,
		    par_dense
		);

	SpikingQifNeuralDynamics dyn(par_ode, par_qif);
	QIFOdeSystem sys(dyn);
	GeomAlgorithm<PopulationConnection> alg_qif(sys);

	DynamicNode<PopulationConnection> node(alg_qif,DynamicLib::EXCITATORY_DIRECT);

	RateAlgorithm<PopulationConnection> alg_e(1.0);
	RateAlgorithm<PopulationConnection> alg_i(1.0);

	DynamicNode<PopulationConnection> node_e(alg_e,DynamicLib::EXCITATORY);
	DynamicNode<PopulationConnection> node_i(alg_i,DynamicLib::INHIBITORY);

	PopulationConnection con_e(1.0,1.0);
	PopulationConnection con_i(1.0,-1.0);

	DynamicNode<PopulationConnection>::connection c_e;
	c_e.first  = &node_e;
	c_e.second = con_e;

	DynamicNode<PopulationConnection>::connection c_i;
	c_i.first = &node_i;
	c_i.second = con_i;

	node.PushBackConnection(c_e);
	node.PushBackConnection(c_i);

	node_e.SetValue(1.0);
	node_i.SetValue(1.0);

	node.CollectExternalInput();

}
*/
//-----------------------------------------------
/*BOOST_AUTO_TEST_CASE(NoInputNetworkTest)
{

	Potential V_min		 = -10.0;
	Potential V_peak	 =  10.0;
	Potential V_reset	 = -10.0;

	Potential V_reversal = 0.0;

	Potential gamma		 = 0.5;
	Time t_mem	       	 = 10e-3;
	Time t_ref	       	 = 0.0;
	QifParameter
		par_qif
		(
			gamma
		);

	Number n_bins = 100;

	PopulationParameter par_neuron(V_peak, V_reset, V_reversal, t_ref, t_mem);

	InitialDensityParameter par_dense(V_min,0.0);
	OdeParameter 
		par_ode
		(
			n_bins,
			V_min,
			par_neuron,
			par_dense
		);

	SpikingQifNeuralDynamics dyn(par_ode, par_qif);
	QifOdeSystem sys(dyn);

	DiffusionParameter par_diff(0.03,0.05);
	CurrentCompensationParameter par_curr(par_qif.Gammasys(), 0.01);
	GeomParameter par_geom(sys,sqrt(gamma), par_diff, par_curr,"NumericalMasterEquation", true);

	GeomAlgorithm<MPILib::populist::OrnsteinUhlenbeckConnection> alg_qif(par_geom);

	Time t_period = dyn.TimeToInf(V_min);
	Time t_step   = t_period/n_bins;

	// the grid_before is empty because no report has been triggered at t = 0;
	// this is normal behaviour
	AlgorithmGrid grid_before = alg_qif.getGrid();

	vector<MPILib::Rate> vec_rate;
	vec_rate.push_back(0.0);
	vector<MPILib::populist::OrnsteinUhlenbeckConnection> vec_con;
	MPILib::populist::OrnsteinUhlenbeckConnection con(1.0, 1.0, 0.0);
	vec_con.push_back(con);

	alg_qif.evolveNodeState(vec_rate, vec_con, t_step);

	AlgorithmGrid grid_after1 = alg_qif.getGrid();
	vector<double> vec_dense1 = grid_after1.toStateVector();


	BOOST_CHECK(vec_dense1[0] == 0);
	BOOST_CHECK(vec_dense1[1] != 0);
	BOOST_CHECK(vec_dense1[2] == 0);

	alg_qif.evolveNodeState(vec_rate, vec_con, 2*t_step);
	AlgorithmGrid grid_after2  = alg_qif.getGrid();
	vector<double> vec_dense2 = grid_after2.toStateVector();

	BOOST_CHECK(vec_dense2[1] == 0);
	BOOST_CHECK(vec_dense2[2] != 0);
	BOOST_CHECK(vec_dense2[3] == 0);
}


//____________________________________________________________________________//

BOOST_AUTO_TEST_CASE(NonStandardResetTest){

	Potential V_min   = -10.0;
	Potential V_peak  =  10.0;
	Potential V_reset =   0.0;
	Potential gamma	  =   0.5;

	Potential V_reversal = 0.0;

	Time t_mem	  =  10e-3;
	Time t_ref	  =   0.0;

	QifParameter
		par_qif
		(
			gamma
		);

	Number n_bins = 100;
	InitialDensityParameter par_dense(V_min,0.0);

	PopulationParameter par_neuron(V_peak, V_reset, V_reversal, t_ref, t_mem);

	OdeParameter 
		par_ode
		(
			n_bins,
			V_min,
			par_neuron,
			par_dense
		);

	DiffusionParameter par_diff(0.03,0.05);
	CurrentCompensationParameter par_curr(par_qif.Gammasys(),0.01);
	SpikingQifNeuralDynamics dyn(par_ode, par_qif);
	QifOdeSystem sys(dyn);
	GeomAlgorithm<MPILib::populist::OrnsteinUhlenbeckConnection> alg_qif(GeomParameter(sys, sqrt(gamma), par_diff, par_curr, "NumericalMasterEquation", true));
	
	vector<MPILib::Rate> vec_rate;
	vec_rate.push_back(0.0);
	vector<MPILib::populist::OrnsteinUhlenbeckConnection> vec_con;
	MPILib::populist::OrnsteinUhlenbeckConnection con(1.0, 1.0, 0.0);
	vec_con.push_back(con);



	Time t_period = dyn.TimeToInf(V_min);
	Time t_step   = t_period/n_bins;
	alg_qif.evolveNodeState(vec_rate, vec_con, (n_bins-1)*t_step);
	AlgorithmGrid grid_before_reset  = alg_qif.getGrid();
	vector<Density> vec_dense = grid_before_reset.toStateVector();
	BOOST_CHECK(vec_dense[n_bins-1] != 0);

	alg_qif.evolveNodeState(vec_rate, vec_con, n_bins*t_step);
	AlgorithmGrid grid_after_reset = alg_qif.getGrid();
	vec_dense = grid_after_reset.toStateVector();
	BOOST_CHECK(vec_dense[51] != 0);
}

//____________________________________________________________________________//


BOOST_AUTO_TEST_CASE(StandardResetTest){

	Potential V_min   = -10.0;
	Potential V_peak  =  10.0;
	Potential V_reset = -10.0;
	Potential gamma	  =   0.5;

	Potential V_reversal = 0.0;

	Time t_mem	  =  10e-3;
	Time t_ref	  =   0.0;

	QifParameter
		par_qif
		(
			gamma
		);

	Number n_bins = 100;
	InitialDensityParameter par_dense(V_min,0.0);

	PopulationParameter par_neuron(V_peak, V_reset, V_reversal, t_ref, t_mem);
	OdeParameter 
		par_ode
		(
			n_bins,
			V_min,
			par_neuron,
			par_dense
		);

	DiffusionParameter par_diff(0.03,0.05);
	CurrentCompensationParameter par_curr(par_qif.Gammasys(),0.01);
	SpikingQifNeuralDynamics dyn(par_ode, par_qif);
	QifOdeSystem sys(dyn);
	GeomAlgorithm<MPILib::populist::OrnsteinUhlenbeckConnection> alg_qif(GeomParameter(sys,sqrt(gamma),par_diff,par_curr,"NumericalMasterEquation",true));
	
	vector<MPILib::Rate> vec_rate;
	vec_rate.push_back(0.0);
	vector<MPILib::populist::OrnsteinUhlenbeckConnection> vec_con;
	MPILib::populist::OrnsteinUhlenbeckConnection con(1.0, 1.0, 0.0);
	vec_con.push_back(con);


	Time t_period = dyn.TimeToInf(V_min);
	Time t_step   = t_period/n_bins;
	alg_qif.evolveNodeState(vec_rate, vec_con, (n_bins-1)*t_step);
	AlgorithmGrid grid_before_reset = alg_qif.getGrid();
	vector<Density> vec_dense = grid_before_reset.toStateVector();
	BOOST_CHECK(vec_dense[n_bins-1] != 0);

	alg_qif.evolveNodeState(vec_rate, vec_con, n_bins*t_step);
	AlgorithmGrid grid_after_reset = alg_qif.getGrid();
	vec_dense = grid_after_reset.toStateVector();

	BOOST_CHECK(vec_dense[1] != 0.0);
}

/*
BOOST_AUTO_TEST_CASE(LIFSingleInputTest){

	Potential V_min	     =  0.0;
	Potential V_peak     =  1.0;
	Potential V_reset    =  0.0;
	Potential V_reversal =  0.0;
	Time t_mem	         =  50e-3;
	Time t_ref	         =  0.0;

	Number n_bins = 300;

	PopulationParameter par_neuron(V_peak, V_reset, V_reversal, t_ref, t_mem);

	InitialDensityParameter par_dense(V_min,0.0);
	OdeParameter
		par_ode
		(
			n_bins,
			V_min,
			par_neuron,
		    par_dense
		);

	LifNeuralDynamics dyn(par_ode, 0.01);
	LeakingOdeSystem sys(dyn);
	GeomAlgorithm<PopulationConnection> alg_geom(sys);

	AlgorithmInputSimulationFixture<OU_Connection> fixt(800.,0.03);
	pair<predecessor_iterator, predecessor_iterator>  par = fixt.GenerateInput();

	Time t_step = sys.TStep();
	alg_geom.EvolveNodeState(par.first, par.second, t_step);

}

*//*
BOOST_AUTO_TEST_CASE(LIFSingleInputTestWithNegativePotential){

	Potential V_min	     = -1.0;
	Potential V_peak     =  1.0;
	Potential V_reset    =  0.0;
	Potential V_reversal =  0.0;
	Time t_mem	         =  50e-3;
	Time t_ref	         =  0.0;

	Number n_bins = 300;

	NeuronParameter par_neuron(V_peak, V_reset, V_reversal, t_ref, t_mem);

	InitialDensityParameter par_dense(V_min,0.0);
	OdeParameter
		par_ode
		(
			n_bins,
			V_min,
			par_neuron,
			par_dense
		);

	LifNeuralDynamics dyn(par_ode, 0.01);
	LeakingOdeSystem sys(dyn);
	GeomAlgorithm<DelayedConnection> alg_geom(sys);

	AlgorithmInputSimulationFixture<OU_Connection> fixt(800.,0.03);
	pair<predecessor_iterator, predecessor_iterator>  par = fixt.GenerateInput();

	Time t_step = sys.TStep();
	alg_geom.EvolveNodeState(par.first, par.second, t_step);

}
//____________________________________________________________________________//


BOOST_AUTO_TEST_SUITE_END()
*/
