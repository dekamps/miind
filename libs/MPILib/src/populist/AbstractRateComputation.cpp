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
#include <MPILib/include/populist/AbstractRateComputation.hpp>
#include <assert.h>
#include <MPILib/include/BasicDefinitions.hpp>

namespace MPILib {
namespace populist {

AbstractRateComputation::AbstractRateComputation() {
}

void AbstractRateComputation::Configure(std::valarray<Density>& array_state,
		const parameters::InputParameterSet& input_set,
		const parameters::PopulationParameter& par_population, Index index_reversal) {
	_p_array_state = &array_state;
	_p_input_set = &input_set;
	_par_population = par_population;
	_index_reversal = index_reversal;
}

AbstractRateComputation::~AbstractRateComputation() {
}

bool AbstractRateComputation::DefineRateArea(Potential v_lower) {

	_delta_v_rel = 1.0 / (_n_bins - 1 - static_cast<double>(_index_reversal));
	_delta_v_abs = (_par_population._theta - _par_population._V_reset)
			* _delta_v_rel;

	// negative values of v_cutoff are allowed, but the result must be an Index;
	_start_integration_area =
			static_cast<Index>(static_cast<int>(_index_reversal)
					+ static_cast<int>(v_lower / _delta_v_rel));

	// if index_start = _n_bins - 1, there is one integration point:
	_number_integration_area = _n_bins - _start_integration_area;

	assert( _start_integration_area < _n_bins);
	// equal sign may occur, rounding errors could lift BinToCurrentPotential to slightly higher than v_cutoff 
	assert(
			BinToCurrentPotential(_start_integration_area) - EPSILON_INTEGRALRATE < v_lower);
	assert( BinToCurrentPotential(_start_integration_area + 1) > v_lower);

	for (int index = _start_integration_area; index < static_cast<int>(_n_bins);
			index++)
		_array_interpretation[index] = BinToCurrentPotential(index);

	return true;
}

Potential AbstractRateComputation::BinToCurrentPotential(Index index) {
	assert(index < _n_bins);
	return (static_cast<int>(index) - static_cast<int>(_index_reversal))
			* _delta_v_rel;
}

} /* namespace populist */
} /* namespace MPILib */
