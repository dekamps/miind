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
#include <assert.h>
#include <MPILib/include/populist/circulantSolvers/AbstractCirculantSolver.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

namespace MPILib {
namespace populist {
namespace circulantSolvers {

AbstractCirculantSolver::AbstractCirculantSolver(CirculantMode mode,
		double precision) :
		_array_rho(MAXIMUM_NUMBER_CIRCULANT_BINS + 1), _array_circulant(
				MAXIMUM_NUMBER_CIRCULANT_BINS + 1), _mode(mode) {
}

void AbstractCirculantSolver::Configure(std::valarray<Potential>* p_array_state,
		const parameters::InputParameterSet& set) {
	_p_array_state = p_array_state;
	_p_set = &set;
	_n_bins = p_array_state->size();

	_initial_integral = p_array_state->sum();
}

Number AbstractCirculantSolver::NrCirculant() const {
	return _p_set->_n_circ_exc;
}

void AbstractCirculantSolver::FillLinear() {
	assert(_p_set->_n_noncirc_exc < MAXIMUM_NUMBER_CIRCULANT_BINS);

#ifndef NDEBUG
	int remainder =
			(_p_set->_H_exc != 0 && (_n_bins) % _p_set->_H_exc == 0) ? 0 : 1;
	unsigned int n_noncirc_exc =
			(_p_set->_H_exc != 0) ? (_n_bins) / _p_set->_H_exc + remainder : 0;
#endif 
	assert(n_noncirc_exc == _p_set->_n_noncirc_exc);

	std::valarray<double> array_state = *_p_array_state;

	for (Index index_non_circulant = 0;
			index_non_circulant < _p_set->_n_noncirc_exc; index_non_circulant++)
		_array_rho[index_non_circulant] = 0;

	for (Index index_density = 0; index_density < _n_bins; index_density++) {
		Index index_non_circulant_area = (_n_bins - 1 - index_density)
				/ (_p_set->_H_exc);

		assert(index_non_circulant_area < static_cast<unsigned int>( _array_rho.size() ));

		_array_rho[index_non_circulant_area] += array_state[index_density];
	}
}

void AbstractCirculantSolver::FillFP() {
	std::valarray<double> array_state = *_p_array_state;
	double h = _p_set->_H_exc + _p_set->_alpha_exc;
	Number n_bounds =
			(floor(_n_bins / h) - _n_bins / h == 0) ?
					static_cast<Number>(floor(_n_bins / h) - 1) :
					static_cast<Number>(floor(_n_bins / h));
	assert(n_bounds + 1 < MAXIMUM_NUMBER_CIRCULANT_BINS);

	double bound = 0;
	for (Index i = 0; i < n_bounds + 1; i++)
		_array_rho[i] = 0;
	int n_start = _n_bins - 1;
	for (Index i = 0; i < n_bounds; i++) {
		bound += h;
		Index bound_index = static_cast<int>(_n_bins)
				- static_cast<int>(floor(bound)) - 1;
		for (Index j = n_start; j > bound_index; j--)
			_array_rho[i] += array_state[j];

		double fract = bound - floor(bound);
		_array_rho[i] += fract * array_state[bound_index];
		_array_rho[i + 1] += (1 - fract) * array_state[bound_index];
		n_start = --bound_index;
	}

	for (int j = n_start; j >= 0; j--)
		_array_rho[n_bounds] += array_state[j];
}

void AbstractCirculantSolver::FillNonCirculantBins() {
	if (_mode == INTEGER)
		this->FillLinear();
	else
		this->FillFP();
}

Density AbstractCirculantSolver::IntegratedFlux() const {
	Density sum = 0;
	for (Index index_circulant = 0; index_circulant < _p_set->_n_circ_exc;
			index_circulant++)
		sum += _array_circulant[index_circulant];

	return sum;
}


double AbstractCirculantSolver::operator [](Index index) const {
	return _array_circulant[index];
}

void AbstractCirculantSolver::AddCirculantToState(Index i_reversal) {
	if (_mode == INTEGER)
		this->AddCirculantInteger(i_reversal);
	else {
		assert(
				_p_set->_n_circ_exc == static_cast<Number>((_n_bins - i_reversal)/(_p_set->_H_exc + _p_set->_alpha_exc))+1);
		this->AddCirculantFP(i_reversal);
	}
}

void AbstractCirculantSolver::AddCirculantInteger(Index i_reversal) {
	std::valarray<double>& array_state = *_p_array_state;
	Index i = i_reversal;
	int n_circ = static_cast<int>(_p_set->_n_circ_exc);
	for (int j = 0; j < n_circ; j++) {
		array_state[i] += _array_circulant[j];
		i += _p_set->_H_exc;
	}
}

void AbstractCirculantSolver::AddCirculantFP(Index i_reversal) {
	std::valarray<double>& array_state = *_p_array_state;
	int n_circ = static_cast<int>(_p_set->_n_circ_exc);
	for (int j = 0; j < n_circ - 1; j++) {
		double h = (_p_set->_H_exc + _p_set->_alpha_exc) * j;
		int H = static_cast<int>(floor(h)) + i_reversal;
		double frac = h - floor(h);
		array_state[H] += (1 - frac) * _array_circulant[j];
		array_state[H + 1] += frac * _array_circulant[j];
	}
	array_state[_n_bins - 1] += _array_circulant[n_circ - 1];
}
} /* namespace circulantSolvers*/
} /* namespace populist */
} /* namespace MPILib */
