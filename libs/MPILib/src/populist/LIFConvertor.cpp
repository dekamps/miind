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
#include <assert.h>
#include <MPILib/include/populist/LIFConvertor.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/populist/MuSigmaScalarProduct.hpp>
#include <MPILib/include/populist/SpecialBins.hpp>
#include <MPILib/include/MPINode.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>

namespace MPILib {
namespace populist {

void LIFConvertor::AdaptParameters() {
	// Purpose: Adaption from stride size to current scale, to be called after every new AddBin, or rebinning
	// Assumes correct values for _delta_v and _set_input, so DeltaV() must be called before this routine
	// Author: Marc de Kamps
	// Date: 21-09-2005

	// Additions: 20-10-2005 One and Two Population modes
	//          : 01-01-2006 separate functions for One and Two Population modes
	//			: 23-03-2009 Move from PopulationGridController to ConvertMuSigmaToH
	//			: 28-07-2011 Now directly responsible for reading the input contribution of other populations. This version now can not interpret diffusion input anymore.

	RecalculateSolverParameters();
}

void LIFConvertor::SetDiffusionParameters(const MuSigma& par) {
	double mu = par._mu;
	double sigma = par._sigma;
	Potential h = sigma * sigma / mu;
	Rate rate = mu * mu / (sigma * sigma * _p_par_pop->_tau);
	if (IsSingleDiffusionProcess(h)) {
		if (h > 0) {
			_input_set._h_exc = h;
			_input_set._h_inh = 0.0;
			_input_set._rate_exc = rate;
			_input_set._rate_inh = 0.0;

		} else {
			_input_set._h_exc = 0.0;
			_input_set._h_inh = -h;
			_input_set._rate_exc = 0.0;
			_input_set._rate_inh = rate;
		}
	} else {
		double h = DIFFUSION_STEP
				* (_p_par_pop->_theta - _p_par_pop->_V_reversal);
		double tau = _p_par_pop->_tau;
		_input_set._h_exc = h;
		_input_set._h_inh = -h;

		_input_set._rate_exc = (sigma * sigma + h * mu) / (2 * h * h * tau);
		_input_set._rate_inh = (sigma * sigma - h * mu) / (2 * h * h * tau);
	}

}

bool LIFConvertor::IsSingleDiffusionProcess(Potential h) const {
	return (h / (_p_par_pop->_theta - _p_par_pop->_V_reversal) < DIFFUSION_LIMIT);
}

void LIFConvertor::UpdateRestInputParameters()
// Purpose: after someone has changed _p_input_set->_H_exc, ..inh, the number
// of non_circulant bins must be adapted
// Author: M. de Kamps
// Date: 26-06-2008
// Modification: 23-03-2009; Moved from PopulationGridController to ConvertMuSigmaToH
{
	int remainder =
			(_input_set._H_exc != 0 && (*_p_n_bins) % _input_set._H_exc == 0) ?
					0 : 1;
	_input_set._n_noncirc_exc =
			(_input_set._H_exc != 0) ?
					(*_p_n_bins) / _input_set._H_exc + remainder : 0;

	remainder =
			(_input_set._H_inh != 0 && (*_p_n_bins) % _input_set._H_inh == 0) ?
					0 : 1;
	_input_set._n_noncirc_inh =
			(_input_set._H_inh != 0) ?
					(*_p_n_bins) / _input_set._H_inh + remainder : 0;

	_input_set._n_circ_exc =
			(_input_set._H_exc != 0) ?
					static_cast<Number>((_p_par_pop->_theta
							- _p_par_pop->_V_reset) / _input_set._h_exc) + 1 :
					_input_set._n_circ_exc = 0;
}
void LIFConvertor::RecalculateSolverParameters() {
	// _delta_v != 0 is guaranteed
	// This step can NOT be moved into UpdateRestParameters, because some versions
	// of the algorithm mess with the values computed here and then call Update....
	_input_set._H_exc = static_cast<int>(floor(
			_input_set._h_exc / (*_p_delta_v)));
	_input_set._H_inh = static_cast<int>(floor(
			-_input_set._h_inh / (*_p_delta_v)));

	// since H are rounded to the next integer, -0.5 <= \alpha 0.5
	_input_set._alpha_exc = _input_set._h_exc / (*_p_delta_v)
			- _input_set._H_exc;
	_input_set._alpha_inh = -_input_set._h_inh / (*_p_delta_v)
			- _input_set._H_inh;

	assert(_input_set._alpha_exc <= 1.0 && _input_set._alpha_exc >= 0.0);
	assert(_input_set._alpha_inh <= 1.0 && _input_set._alpha_inh >= 0.0);

	UpdateRestInputParameters();
}

void LIFConvertor::Configure(std::valarray<Potential>& array_state) {
}

const Index& LIFConvertor::IndexReversalBin() const {
	return _p_bins->_index_reversal_bin;
}

const Index& LIFConvertor::IndexCurrentResetBin() const {
	return _p_bins->_index_current_reset_bin;
}

void LIFConvertor::SortConnectionvector(const std::vector<Rate>& nodeVector,
		const std::vector<OrnsteinUhlenbeckConnection>& weightVector) {

	// sorting depends on network structure and only should be done once
//	typedef DynamicLib::DynamicNode<PopulationConnection>& Node;
	if (!_b_toggle_sort) {
		auto iterWeight= weightVector.begin();
		for (auto iter = nodeVector.begin(); iter != nodeVector.end(); iter++, iterWeight++) {
//			AbstractSparseNode<double,PopulationConnection>& sparse_node = *iter;
			MPINode<PopulationConnection, utilities::CircularDistribution> node =
					*iter;

			if (node.getNodeType() == EXCITATORY_BURST
					|| node.getNodeType() == INHIBITORY_BURST)
				_vec_burst.push_back(iter);
			else
				_vec_diffusion.push_back(iter);
		}
		_b_toggle_sort = true;
	}
	if (_vec_burst.size() == 1 && _vec_diffusion.size() == 0) {
		double h = iter_begin.GetWeight()._efficacy;
		double rate = iter_begin->GetValue();
		if (h >= 0) {
			_input_set._h_exc = h;
			_input_set._h_inh = 0;
			_input_set._rate_exc = rate;
			_input_set._rate_inh = 0.0;
		} else {
			_input_set._h_exc = 0;
			_input_set._h_inh = h;
			_input_set._rate_exc = 0.0;
			_input_set._rate_inh = rate;
		}

		iter_begin++;
		// one and only one input
	}

	if (_vec_burst.size() == 0 && _vec_diffusion.size() > 0) {
		MuSigmaScalarProduct scalar;
		MuSigma par = scalar.Evaluate(nodeVector, weightVector, _p_par_pop->_tau);
		SetDiffusionParameters(par);
	}
}

} /* namespace populist */
} /* namespace MPILib */
