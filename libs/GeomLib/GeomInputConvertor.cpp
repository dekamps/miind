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
#include "../MPILib/include/algorithm/AlgorithmInterface.hpp"
#include "GeomInputConvertor.hpp"
#include "MuSigmaScalarProduct.hpp"
#include "OrnsteinUhlenbeckParameter.hpp"
#include "GeomLibException.hpp"


using namespace GeomLib;

GeomInputConvertor::GeomInputConvertor
(
	const OrnsteinUhlenbeckParameter&		par_neuron,
	const DiffusionParameter&				par_diff,
	const CurrentCompensationParameter&		par_curr,
	const std::vector<MPILib::Potential>&	vec_int,
	bool									b_force_small
):
_par_neuron			(par_neuron),
_par_diff			(par_diff),
_par_curr			(par_curr),
_vec_interpretation	(vec_int),
_vec_direct			(0),
_vec_diffusion		(0),
_force_small_bins	(b_force_small)
{
}

MPILib::Number GeomInputConvertor::NumberDirect() const
{
	return _vec_direct.size();
}

void GeomInputConvertor::SetDiffusionParameters
(
	const MuSigma& par,
	InputParameterSet& set
) const
{
	double mu    = par._mu;
	double sigma = par._sigma;

	if (mu == 0.0 && sigma ==0.0){
		set._h_exc = 0.0;
		set._h_inh = 0.0;
		set._rate_exc = 0.0;
		set._rate_inh = 0.0;
		return;
	}

	MPILib::Potential h = (mu != 0.0) ? sigma*sigma/mu : std::numeric_limits<double>::max();

	if (mu != 0 && IsSingleDiffusionProcess(h) ){

		if (fabs(h) < 2*_min_step && _force_small_bins)
			throw GeomLibException("Increase the number of bins.");

		MPILib::Rate rate = mu*mu/(sigma*sigma*_par_neuron._tau);

		if ( h > 0 ){
			set._h_exc    = h;
			set._h_inh    = 0.0;
			set._rate_exc = rate;
			set._rate_inh = 0.0;
		}
		else {
			set._h_exc	  = 0.0;
			set._h_inh	  =  h; // removed - sign, MdK 04/08/2012; the step size need to be forwarded as a negative number
			set._rate_exc = 0.0;
			set._rate_inh = rate;
		}
	}
	else {
		double h = _par_diff._diffusion_jump*(_par_neuron._theta - _vec_interpretation[0]);

		if (fabs(h) < 2*_min_step && _force_small_bins)
			throw GeomLibException("Increase the number of bins.");

		double tau =  _par_neuron._tau;
		set._h_exc =  h;
		set._h_inh = -h;

		set._rate_exc = (sigma*sigma + h*mu)/(2*h*h*tau);
		set._rate_inh = (sigma*sigma - h*mu)/(2*h*h*tau);

		assert( set._rate_exc >= 0.0);
		assert( set._rate_inh >= 0.0);
	}
}

bool GeomInputConvertor::IsSingleDiffusionProcess(MPILib::Potential h) const
{
	return (fabs(h) < _par_diff._diffusion_limit*(_par_neuron._theta - _vec_interpretation[0]));
}

void GeomInputConvertor::SortConnectionvector
(
	const std::vector<MPILib::Rate>& 									vec_rates,
	const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>& 	vec_con,
	const std::vector<MPILib::NodeType>& 								vec_type
)
{
	assert(vec_rates.size() == vec_con.size());
	assert(vec_type.size()  == vec_rates.size());

	// it is guaranteed that there is exist a parameter vector that is large enough
	if (_vec_set.size() == 0)
		// If all inputs are direct, still one extra input is needed for diffusion
		_vec_set = std::vector<InputParameterSet>(vec_type.size() + 1);

	_vec_direct.clear();
	_vec_diffusion.clear();

	for (MPILib::Index i = 0; i < vec_type.size(); i++)
		if (vec_type[i] == MPILib::EXCITATORY_GAUSSIAN ||
		    vec_type[i] == MPILib::INHIBITORY_GAUSSIAN)
			_vec_diffusion.push_back(i);
		else
			_vec_direct.push_back(i);

	this->AddDiffusionParameter (vec_con, vec_rates);
	this->AddBurstParameters    (vec_con, vec_rates);
}

void GeomInputConvertor::AddBurstParameters
(
	const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>& vec_con,
	const std::vector<MPILib::Rate>& vec_rates
)
{
	MPILib::Index start = 1;
	for(auto i: _vec_direct){
		
		double h = vec_con[i]._efficacy;
		double N = vec_con[i]._number_of_connections;
		double rate = vec_rates[i];

		if (h >= 0 ){
			_vec_set[start]._h_exc    = h;
			_vec_set[start]._h_inh    = 0;
			_vec_set[start]._rate_exc = rate*N;
			_vec_set[start]._rate_inh = 0.0;
		} else
		{
			_vec_set[start]._h_exc	 = 0;
			_vec_set[start]._h_inh	 = h;
			_vec_set[start]._rate_exc = 0.0;
			_vec_set[start]._rate_inh = rate*N;
		}
		start++;
	}
}

void GeomInputConvertor::SortDiffusionInput
(
	const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>& vec_con,
	const std::vector<MPILib::Rate>& vec_rates,
	std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>* p_vec_con_diff,
	std::vector<MPILib::Rate>* p_vec_rates_diff
)
{
	for (auto i: _vec_diffusion){
		p_vec_con_diff->push_back(vec_con[i]);
		p_vec_rates_diff->push_back(vec_rates[i]);
	}
}


void GeomInputConvertor::AddDiffusionParameter
(
	const std::vector<MPILib::populist::OrnsteinUhlenbeckConnection>& vec_con,
	const std::vector<MPILib::Rate>& vec_rates
)
{	
	std::vector<MPILib::populist::OrnsteinUhlenbeckConnection> vec_diff_con;
	std::vector<MPILib::Rate> vec_diff_rates;

	SortDiffusionInput(vec_con,vec_rates, &vec_diff_con,&vec_diff_rates);

	MuSigmaScalarProduct scalar;
	MuSigma par = 
		scalar.Evaluate
		(
			vec_diff_rates,
			vec_diff_con,
			_par_neuron._tau
		);

	par._mu    += _par_curr._I;
	par._sigma += _par_curr._sigma;

	SetDiffusionParameters(par,_vec_set[0]);
}

MPILib::Potential GeomInputConvertor::MinVal
(
	const std::vector<MPILib::Potential>& vec_interpretation
) const
{
	assert (vec_interpretation.size() > 0);
	double min = std::numeric_limits<double>::max();
	for (MPILib::Index i = 0; i < vec_interpretation.size() - 1; i++){
		double dif = fabs(vec_interpretation[i+1] - vec_interpretation[i]);
		if (min > dif)
			min = dif;
	}
	return min;
}

