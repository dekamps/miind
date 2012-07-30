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
#include <MPILib/include/populist/circulantSolvers/PolynomialCirculant.hpp>

namespace MPILib {
namespace populist {
namespace circulantSolvers {

PolynomialCirculant::PolynomialCirculant() :
		_j_array(std::vector<double>(CIRCULANT_POLY_JMAX)) {
}


PolynomialCirculant* PolynomialCirculant::clone() const {
	return new PolynomialCirculant;
}

void PolynomialCirculant::Execute(Number n_bins, Time tau, Time t_irrelevant) {
	assert( _p_set->_n_circ_exc < MAXIMUM_NUMBER_GAMMAZ_VALUES);
	assert( _p_set->_n_noncirc_exc < MAXIMUM_NUMBER_GAMMAZ_VALUES);

	_n_bins = n_bins;
	_tau = tau;

	FillNonCirculantBins();

	LoadJArray();

	for (Index i_circulant = 0; i_circulant < CIRCULANT_POLY_DEGREE;
			i_circulant++) {
		_array_circulant[i_circulant] = 0;
		for (Index i_non_circulant = 0;
				i_non_circulant < CIRCULANT_POLY_DEGREE - i_circulant;
				i_non_circulant++) {
			_array_circulant[i_circulant] += _j_array[i_circulant
					+ i_non_circulant] * _array_rho[i_non_circulant];

		}
	}
}

Number PolynomialCirculant::NrCirculant() const {
	return CIRCULANT_POLY_DEGREE;
}

Index PolynomialCirculant::JMax() const {
	return CIRCULANT_POLY_JMAX;
}

void PolynomialCirculant::LoadJArray() {
	double sum = 1.0;
	double fac_p = 1.0;

	for (Index p = 1; p <= CIRCULANT_POLY_DEGREE; p++) {
		fac_p *= -_tau / static_cast<double>(p);
		sum += fac_p;
	}

	double fac_j = 1.0;

	for (Index j = 1; j <= CIRCULANT_POLY_JMAX; j++) {
		fac_j *= _tau / static_cast<double>(j);
		_j_array[j - 1] = fac_j * sum;
	}
}

void PolynomialCirculant::FillNonCirculantBins() {
	// Purpose: Integrate the density in the non-circulant areas. This produces the vector f^0.
	// Assumptions: none
	// Author: Marc de Kamps
	// Date:   01-09-2005
	// Modified: 10-07-2006; Insert break after CIRCULANT_POLY_J_MAX

	std::valarray<double> array_state = *_p_array_state;

	for (Index index_non_circulant = 0;
			index_non_circulant < _p_set->_n_noncirc_exc; index_non_circulant++)
		_array_rho[index_non_circulant] = 0;

	for (int index_density = _n_bins - 1; index_density >= 0; index_density--) {
		Index index_non_circulant_area = (_n_bins - 1 - index_density)
				/ (_p_set->_H_exc);

		if (index_non_circulant_area > CIRCULANT_POLY_JMAX)
			break;

		assert(
				index_non_circulant_area >= 0 && index_non_circulant_area < static_cast<unsigned int>( _array_rho.size() ));

		_array_rho[index_non_circulant_area] += array_state[index_density];
	}
}
} /* namespace circulantSolvers*/
} /* namespace populist */
} /* namespace MPILib */
