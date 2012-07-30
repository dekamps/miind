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
#include <gsl/gsl_math.h>
#include <MPILib/include/populist/circulantSolvers/CirculantSolver.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include <iostream>

namespace MPILib {
namespace populist {
namespace circulantSolvers {

CirculantSolver::CirculantSolver(CirculantMode mode) :
		AbstractCirculantSolver(mode) {
}

void CirculantSolver::Execute(Number n_bins, Time tau, Time // absolute simulation time, not required for this circulant
		) {
	assert( _p_set->_n_circ_exc < MAXIMUM_NUMBER_CIRCULANT_BINS);
	assert( _p_set->_n_noncirc_exc < MAXIMUM_NUMBER_NON_CIRCULANT_BINS);

	_n_bins = n_bins;
	_tau = tau;

	FillNonCirculantBins();

	CalculateInnerProduct();
}

void CirculantSolver::CalculateInnerProduct() {
	_array_V.FillArray(_p_set->_n_circ_exc, _p_set->_n_noncirc_exc, _tau);

	for (Index i_circulant = 0; i_circulant < _p_set->_n_circ_exc;
			i_circulant++) {
		_array_circulant[i_circulant] = 0;
		for (Index i_non_circulant = 0;
				i_non_circulant < _p_set->_n_noncirc_exc; i_non_circulant++) {
			// Here the circulant solution is applied and the values of the circulant solution are stored
			// in _array_circulant
			_array_circulant[i_circulant] += _array_V.V(i_circulant,
					i_non_circulant) * _array_rho[i_non_circulant];

		}
	}
}

double CirculantSolver::Flux(Number number_non_circulant_areas,
		Time tau) const {
	//TODO: fill in proper reference
	// Author: Marc de Kamps
	// Date: 01-09-2005

	assert(number_non_circulant_areas < MAXIMUM_NUMBER_CIRCULANT_BINS);
	double sum = _array_rho[0];
	double fact = tau;
	for (int i = 1; i < static_cast<int>(number_non_circulant_areas); i++) {
		sum += fact * _array_rho[i];
		fact *= tau / static_cast<double>(i);
	}
	sum *= exp(-tau);
	return sum;
}

double CirculantSolver::Integrate(Number number_circulant) const {

	double f_return = 0;
	for (int index_sum = 0; index_sum < static_cast<int>(number_circulant);
			index_sum++)
		f_return += _array_circulant[index_sum];

	return f_return;
}

CirculantSolver* CirculantSolver::clone() const {
	return new CirculantSolver;
}
} /* namespace circulantSolvers*/
} /* namespace populist */
} /* namespace MPILib */
