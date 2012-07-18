// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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

#include <MPILib/include/populist/ABConvertor.hpp>
#include <MPILib/include/populist/AbstractCirculantSolver.hpp>
#include <MPILib/include/populist/parameters/PopulistSpecificParameter.hpp>

namespace MPILib {
namespace populist {

ABConvertor::ABConvertor( VALUE_REF_INIT
SpecialBins&, PopulationParameter& par_pop,
		PopulistSpecificParameter& par_specific, Potential& delta_v,
		Number& n_current_bins) :
		VALUE_MEMBER_INIT
		_p_specific(&par_specific), _p_pop(&par_pop), _p_n_bins(
				&n_current_bins), _p_delta_v(&delta_v) {
}

const PopulistSpecificParameter&
ABConvertor::PopSpecific() const {
	return *_p_specific;
}

const OneDMInputSetParameter&
ABConvertor::InputSet() const {
	return _param_input;
}

void ABConvertor::SortConnectionvector(const std::vector<Rate>& nodeVector,
		const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
		const std::vector<NodeType>& typeVector) {
	_param_input._par_input = _scalar_product.Evaluate(nodeVector, weightVector,
			_p_pop->_tau);
	_param_input._par_input._q = _param_onedm._par_adapt._q;
}

void ABConvertor::AdaptParameters(

) {

	RecalculateSolverParameters();
}

void ABConvertor::RecalculateSolverParameters() {
	_param_input._n_current_bins = *_p_n_bins;
	_param_input._n_max_bins = _p_specific->getMaxNumGridPoints();

	// current expansion factor is current number of bins
	// divided by number of initial bins

	double f = static_cast<double>(_param_input._n_current_bins)
			/ static_cast<double>(_p_specific->getNrGridInitial());
	_param_input._q_expanded = f * _param_input._par_input._q;
	_param_input._t_since_rebinning = _p_pop->_tau * log(f);
	_param_input._g_max = _p_pop->_theta;
	_param_input._tau = _p_pop->_tau;
}

} /* namespace populist */
} /* namespace MPILib */
