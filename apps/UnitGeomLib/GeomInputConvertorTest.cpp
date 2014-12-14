
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
#include <MPILib/include/NodeType.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckConnection.hpp>

using namespace GeomLib;

struct ConvertorFixture {

	ConvertorFixture();
	~ConvertorFixture(){}

	std::vector<InputParameterSet>
		GenerateSet
		(
			const std::vector<MPILib::Rate>&,
			const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>&,
			const std::vector<MPILib::NodeType>&
		);

	OrnsteinUhlenbeckParameter		_par_neuron;
	DiffusionParameter				_par_diff;
	CurrentCompensationParameter	_par_curr;
	OdeParameter					_par_ode;
	LifNeuralDynamics				_dyn;

	std::unique_ptr<GeomInputConvertor> _conv;
};

std::vector<InputParameterSet> ConvertorFixture::GenerateSet
(
	const std::vector<MPILib::Rate>&                                  vec_rate,
	const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>& vec_con,
	const std::vector<MPILib::NodeType>&							  vec_type
)
{
	_conv = std::unique_ptr<GeomInputConvertor>(new GeomInputConvertor(_par_neuron,_par_diff,_par_curr,_dyn.InterpretationArray()));
	_conv->SortConnectionvector(vec_rate,vec_con,vec_type);
	return _conv->SolverParameter();
}

ConvertorFixture::ConvertorFixture():
_par_neuron(20e-3,0.0,0.0,0.,10e-3),
_par_diff(0.03,0.05),
_par_curr(0.0,0.),
_par_ode(5,0.0,_par_neuron,InitialDensityParameter(0.,0.)),
_dyn(_par_ode,0.01)
{
}

BOOST_AUTO_TEST_CASE(SingleInputTest) {

	vector<MPILib::NodeType> 	vec_type;
	vector<MPILib::Rate>     	vec_rate;
	vector<MPILib::populist::OrnsteinUhlenbeckConnection>	vec_con;

	OrnsteinUhlenbeckParameter par_neuron(20e-3,0.0,0.0,0.,10e-3);
	DiffusionParameter par_diff(0.03,0.05);

	CurrentCompensationParameter par_curr;
	InitialDensityParameter par_dense(0.,0.);

	Number n_bins = 5;
	Potential V_min = 0.0;

	OdeParameter par_ode(n_bins,V_min,par_neuron,par_dense);
	LifNeuralDynamics dyn(par_ode,0.01);
	vector<Potential> vec_interp = dyn.InterpretationArray();

	GeomInputConvertor
		convertor(
			par_neuron,
			par_diff,
			par_curr,
			vec_interp
		);

	vec_type.push_back(MPILib::EXCITATORY_DIRECT);
	MPILib::Rate rate = 800.0;
	vec_rate.push_back(rate);

	MPILib::Efficacy eff = 0.03;
	MPILib::populist::OrnsteinUhlenbeckConnection con(1,eff,0.0);
	vec_con.push_back(con);

	convertor.SortConnectionvector(vec_rate, vec_con, vec_type);
	std::vector<InputParameterSet> vec_set = convertor.SolverParameter();

	BOOST_CHECK(vec_set.size() == 2);
	BOOST_CHECK(vec_set[0]._rate_exc  == 0.0);
	BOOST_CHECK(vec_set[0]._rate_inh  == 0.0);
	BOOST_CHECK(vec_set[1]._rate_exc == rate);
	BOOST_CHECK(vec_set[1]._h_exc    == eff);
	BOOST_CHECK(vec_set[1]._rate_inh == 0.0);
	BOOST_CHECK(vec_set[1]._h_inh    == 0.0);

}

BOOST_FIXTURE_TEST_CASE(test_fixture, ConvertorFixture)
{
	// Repeat the last test to examine the fixture

	vector<MPILib::NodeType> 	vec_type;
	vector<MPILib::Rate>     	vec_rate;
	vector<MPILib::populist::OrnsteinUhlenbeckConnection>	vec_con;

	vec_type.push_back(MPILib::EXCITATORY_DIRECT);
	MPILib::Rate rate = 800.0;
	vec_rate.push_back(rate);

	MPILib::Efficacy eff = 0.03;
	MPILib::populist::OrnsteinUhlenbeckConnection con(1,eff,0.0);
	vec_con.push_back(con);


	std::vector<InputParameterSet> vec_set = GenerateSet(vec_rate, vec_con, vec_type);


	BOOST_CHECK(vec_set.size() == 2);
	BOOST_CHECK(vec_set[0]._rate_exc  == 0.0);
	BOOST_CHECK(vec_set[0]._rate_inh  == 0.0);
	BOOST_CHECK(vec_set[1]._rate_exc == rate);
	BOOST_CHECK(vec_set[1]._h_exc    == eff);
	BOOST_CHECK(vec_set[1]._rate_inh == 0.0);
	BOOST_CHECK(vec_set[1]._h_inh    == 0.0);

}


BOOST_FIXTURE_TEST_CASE(test_current_compensation, ConvertorFixture)
{
	// Repeat the last test to examine the fixture

	vector<MPILib::NodeType> 	vec_type;
	vector<MPILib::Rate>     	vec_rate;
	vector<MPILib::populist::OrnsteinUhlenbeckConnection>	vec_con;

	vec_type.push_back(MPILib::EXCITATORY_DIRECT);
	MPILib::Rate rate = 800.0;
	vec_rate.push_back(rate);

	MPILib::Efficacy eff = 0.03;
	MPILib::populist::OrnsteinUhlenbeckConnection con(1,eff,0.0);
	vec_con.push_back(con);

	double mu    = 0.0;
	double sigma = 0.01;
	_par_curr = CurrentCompensationParameter(mu,sigma);

	std::vector<InputParameterSet> vec_set = GenerateSet(vec_rate, vec_con, vec_type);

	BOOST_CHECK(vec_set.size() == 2);
	BOOST_CHECK(vec_set[1]._rate_exc == rate);
	BOOST_CHECK(vec_set[1]._h_exc    == eff);
	BOOST_CHECK(vec_set[1]._rate_inh == 0.0);
	BOOST_CHECK(vec_set[1]._h_inh    == 0.0);

	double h_e  = vec_set[0]._h_exc;
	double h_i  = vec_set[0]._h_inh;
	double nu_e = vec_set[0]._rate_exc;
	double nu_i = vec_set[0]._rate_inh;

	BOOST_CHECK(h_e == -h_i);

	double mu_r = _par_neuron._tau*(nu_e*h_e + nu_i*h_i);
	double sigma_r = sqrt(_par_neuron._tau*(nu_e*h_e*h_e + nu_i*h_i*h_i));

	BOOST_CHECK(mu == mu_r);
	BOOST_CHECK(sigma == sigma_r);

}

BOOST_FIXTURE_TEST_CASE(test_double_diffusion, ConvertorFixture)
{
	// Repeat the last test to examine the fixture

	vector<MPILib::NodeType> 	vec_type;
	vector<MPILib::Rate>     	vec_rate;
	vector<MPILib::populist::OrnsteinUhlenbeckConnection>	vec_con;

	vec_type.push_back(MPILib::EXCITATORY_GAUSSIAN);
	vec_type.push_back(MPILib::INHIBITORY_GAUSSIAN);

	MPILib::Rate rate = 800.0;
	vec_rate.push_back(rate);
	vec_rate.push_back(rate);

	MPILib::Efficacy eff = 0.01;
	MPILib::populist::OrnsteinUhlenbeckConnection con_e(1,eff,0.0);
	vec_con.push_back(con_e);
	MPILib::populist::OrnsteinUhlenbeckConnection con_i(1,-eff,0.0);
	vec_con.push_back(con_i);

	std::vector<InputParameterSet> vec_set = GenerateSet(vec_rate, vec_con, vec_type);

	BOOST_CHECK(vec_set.size() == 3);             // always one more than there are inputs
	BOOST_CHECK(_conv->NumberDirect() == 0);  // but the number of direct inputs should be 0

	double h_e  = vec_set[0]._h_exc;
	double h_i  = vec_set[0]._h_inh;
	double nu_e = vec_set[0]._rate_exc;
	double nu_i = vec_set[0]._rate_inh;

	double mu     = _par_neuron._tau*(h_e*nu_e + h_i*nu_i);
	double sigma  = sqrt(_par_neuron._tau*(h_e*h_e*nu_e + h_i*h_i*nu_i));

	BOOST_CHECK(mu == 0.0);
	double sigma_r = sqrt(_par_neuron._tau*(2*eff*eff*rate));
	BOOST_CHECK(sigma_r  ==  sigma);
}

