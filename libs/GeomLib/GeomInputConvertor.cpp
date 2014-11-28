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
//#include "SpecialBins.h"
/*
using namespace GeomLib;

		CharacteristicInputConvertor::CharacteristicInputConvertor
		(
			Time						tau,
			const vector<Potential>&	vec_interpretation,
			const vector<Density>&		vec_density,
			Potential					dc_component,
			double						diffusion_limit,
			double						diffusion_step,
			double						sigma_fraction,
			bool						no_sigma_smooth,
			bool						force_small_bins
		):
		_vec_interpretation(vec_interpretation),
		_vec_set(1),		// many older algorithms, OldLIFZeroLeakEquations, SingleInputZeroLeakEquations simply assume there is one InputSetParameter
		_gamma_estimated(EstimateGamma(vec_interpretation)),
		_minval(MinVal(vec_interpretation)),
		_V_min(V_min_from_gamma(_gamma_estimated)),
		_V_max(V_max_from_gamma(_gamma_estimated)),
		_tau(tau),
		_n_bins(vec_interpretation.size()),
		_b_toggle_sort(false),
		_b_toggle_diffusion(false),
		_diffusion_limit(diffusion_limit),
		_diffusion_step(diffusion_step),
		_dc_component(dc_component),
		_sigma_fraction(sigma_fraction),
		_no_sigma_smooth(no_sigma_smooth),
		_force_small_bins(force_small_bins),
		_vec_burst(0),
		_vec_diffusion(0)
		{
			// warn against possible negative firing rates
			if (_diffusion_step > _diffusion_limit)
				throw PopulistException("Keep diffusion_step smaller than diffusion limit");
			predecessor_iterator iter = DCComponent(_dc_component); 
			_vec_diffusion.push_back(iter);
		}

void CharacteristicInputConvertor::AdaptParameters
(
)
{
	RecalculateSolverParameters();
	UpdateRestInputParameters();
}

void CharacteristicInputConvertor::AdaptSigma(Potential mu, Potential* p_h, Potential* p_sigma) const
{
	assert (*p_sigma > 0.0);
	//sigma is so small that it leads to an h value that doesn't even cover a single bin
	// just increase sigma a little bit, but throw an exception if it crosses
	// _sigma_fraction. //TODO: provide a warning in the log file that this has happened
	*p_sigma *= 1.1;
	*p_h = (mu != 0.0) ? (*p_sigma)*(*p_sigma)/mu : numeric_limits<double>::max();
	if (*p_sigma > fabs(mu)*_sigma_fraction )
		throw PopulistException("Can't adapt sigma. Increase number of bins");
}

void CharacteristicInputConvertor::SetDiffusionParameters
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
	Potential h = (mu != 0.0) ? sigma*sigma/mu : numeric_limits<double>::max();

	if (mu != 0 && IsSingleDiffusionProcess(h) ){
		if (fabs(h) < 2*_minval && _force_small_bins)
			throw PopulistException("Increase the number of bins.");

		Rate rate = mu*mu/(sigma*sigma*_tau);

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

		double h = this->DiffusionJump();

		if (fabs(h) < 2*_minval && _force_small_bins)
			throw PopulistException("Increase the number of bins.");

		double tau = _tau;
		set._h_exc =  h;
		set._h_inh = -h;

		set._rate_exc = (sigma*sigma + h*mu)/(2*h*h*tau);
		set._rate_inh = (sigma*sigma - h*mu)/(2*h*h*tau);

		assert( set._rate_exc >= 0.0);
		assert( set._rate_inh >= 0.0);
	}
}

bool CharacteristicInputConvertor::IsSingleDiffusionProcess(Potential h) const
{
	return (fabs(h) < _diffusion_limit*(_V_max - _V_min));
}

void CharacteristicInputConvertor::UpdateRestInputParameters
(
)
{
	BOOST_FOREACH(InputParameterSet& set, _vec_set){
		set._n_noncirc_exc  = 0;
		set._n_noncirc_inh  = 0;
		set._n_circ_exc     = 0;
	}
}

double CharacteristicInputConvertor::EstimateGamma(const vector<double>& vec_interpretation) const
{
	double f = numeric_limits<double>::max();
	Index ind_min = 0;
	for (Index i = 0; i < vec_interpretation.size(); i++)
		if (fabs(vec_interpretation[i]) < f ){
			ind_min = i;
			f = fabs(vec_interpretation[i]);
		}

	Potential V1 = vec_interpretation[ind_min + 1];
	Potential V2 = vec_interpretation[ind_min + 2];
	return V1*V1*V2/(V2 - 2*V1);
}

void CharacteristicInputConvertor::RecalculateSolverParameters
( 
)
{
	// We will not use an expression in an integer number of steps unlike LIFConvertor

	BOOST_FOREACH(InputParameterSet& set, _vec_set){
		set._H_exc = 0;
		set._H_inh = 0;

		assert(set._H_exc >= 0);
		assert(set._H_inh >= 0);

		set._alpha_exc = 0.0;
		set._alpha_inh = 0.0;

		assert(set._alpha_exc <= 1.0 && set._alpha_exc >= 0.0);
		assert(set._alpha_inh <= 1.0 && set._alpha_inh >= 0.0);
	}
}

void CharacteristicInputConvertor::Configure
(
	valarray<Potential>& array_state
)
{
}

void CharacteristicInputConvertor::SortConnectionvector
(
//	predecessor_iterator iter_begin,
//	predecessor_iterator iter_end
		const vector<Rate>&,
		const vector<OrnsteinUhlenbeckConnection>&,
		const vector<MPILib::NodeType>&
)
{
	assert(iter_begin != iter_end);

	// sorting depends on network structure and only should be done once
	typedef DynamicLib::DynamicNode<PopulationConnection>& Node;
	if (! _b_toggle_sort){
		for (predecessor_iterator iter = iter_begin; iter != iter_end; iter++)
		{	
			AbstractSparseNode<double,PopulationConnection>& sparse_node = *iter;
			Node node = dynamic_cast<Node>(sparse_node);

			if ( node.Type() == DynamicLib::EXCITATORY_BURST || node.Type() == DynamicLib::INHIBITORY_BURST )
				_vec_burst.push_back(iter);
			else
				_vec_diffusion.push_back(iter);
		}
		_b_toggle_sort = true;
	}

	this->AddDiffusionParameter();
	this->AddBurstParameters();
}

void CharacteristicInputConvertor::AddBurstParameters()
{
	// This is necessary for older ZeroLeakEquations. They rely on the relevant input 
	// being in _vec-set[0]. So if there is no diffusion input, the first element has
	// to be taken up by bursting input.

	if (_vec_diffusion.size() == 0 )
		_vec_set.clear();

	InputParameterSet set;
	while( _vec_set.size() < 1 + _vec_burst.size() )
		_vec_set.push_back(set);

	Index start = _vec_diffusion.size() == 1 ? 1 : 0;
	BOOST_FOREACH(predecessor_iterator iter, _vec_burst){
		
		double h = iter.GetWeight()._efficacy;
		double N = iter.GetWeight()._number_of_connections;
		double rate = iter->GetValue();

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

void CharacteristicInputConvertor::AddDiffusionParameter()
{	
	MuSigmaScalarProduct scalar;
	MuSigma par = 
		scalar.Evaluate
		(
			_vec_diffusion,
			_tau,
			_no_sigma_smooth
		);
	SetDiffusionParameters(par,_vec_set[0]);
}

AbstractSparseNode<double,OrnsteinUhlenbeckConnection>::predecessor_iterator
	CharacteristicInputConvertor::DCComponent(Potential I)
{
	_connection.first  = &_node;
	OU_Connection conou;
	conou._delay = 0.0;

	if (I == 0){
		_node.SetValue(0);
		conou._efficacy = 0.0;
		conou._number_of_connections = 0;
		_connection.second = conou;
	} else {
		Potential mu = -I;
		Potential sigma = this->_sigma_fraction;

		Rate rate = mu*mu/(_tau*sigma*sigma);
		Potential h = sigma*sigma/mu;
		_node.SetValue(rate);
		conou._efficacy = h;
		conou._number_of_connections = 1;
		_connection.second = conou;
	}
		
	predecessor_iterator iter(&_connection);

	return iter;
}

double CharacteristicInputConvertor::MinVal(const vector<Potential>& vec_interpretation) const
{
	assert (vec_interpretation.size() > 0);
	double min = numeric_limits<double>::max();
	for (Index i = 0; i < vec_interpretation.size() - 1; i++){
		double dif = fabs(vec_interpretation[i+1] - vec_interpretation[i]);
		if (min > dif)
			min = dif;
	}
	return min;
}

Potential CharacteristicInputConvertor::V_min_from_gamma(Potential gamma) const
{
	// for some neuron models V_min is a huge underestimate of the relevant potential domain
	// if gamma can be estimated, use -sqrt(gamma) as minimum potential
	if (gamma > 0)
		return -sqrt(gamma);
	else
		return _vec_interpretation[0];
}

Potential CharacteristicInputConvertor::V_max_from_gamma(Potential gamma) const
{
	if (gamma > 0)
		return sqrt(gamma);
	else
		return _vec_interpretation.back();
}
*/
