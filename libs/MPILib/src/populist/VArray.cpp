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
#ifdef WIN32
#pragma warning(disable: 4996)
#endif

#include <cassert>
#include <functional>
#include <gsl/gsl_math.h>
#include <MPILib/include/populist/VArray.hpp>
#include <MPILib/include/BasicDefinitions.hpp>


namespace MPILib {
namespace populist {

VArray::VArray() :
		_vector_array(MAX_V_ARRAY) {
}

bool VArray::FillArray(Number number_circulant_bins,
		Number number_non_circulant_areas, Time tau) {
	assert(FillArrayWithGarbage());
	assert(CheckInNumbers(number_circulant_bins, number_non_circulant_areas));
	assert(
			number_circulant_bins + number_non_circulant_areas + 1 < _vector_array.size());

	_number_of_circulant_bins = number_circulant_bins;
	_number_of_non_circulant_areas = number_non_circulant_areas;

	double emintau = exp(-tau);
	// first fill the circulant bins V_k 0

	for (int index_circulant = 0;
			index_circulant < static_cast<int>(number_circulant_bins);
			index_circulant++) {

		_vector_array[index_circulant] = (1 - emintau);

		// if n is even, add 
		if (number_circulant_bins % 2 == 0) {
			double sign = (index_circulant % 2) ? -1.0 : 1.0;
			_vector_array[index_circulant] += sign
					* (emintau - emintau * emintau);
		}

		int l_max = number_circulant_bins / 2 - (number_circulant_bins - 1) % 2;

		for (int l = 1; l <= l_max; l++) {
			double c_l = cos(
					2 * M_PI * l / static_cast<double>(number_circulant_bins));
			double s_l = sin(
					2 * M_PI * l / static_cast<double>(number_circulant_bins));

			double arg = 2 * M_PI * l * (index_circulant + 1)
					/ static_cast<double>(number_circulant_bins);
			double arg_cos = s_l * tau - arg;

			_vector_array[index_circulant] += 2 * emintau
					* (exp(c_l * tau) * cos(arg_cos) - cos(arg));

		}
	}

	double factor = tau;
	double J = 1.0;

	for (int index_k_plus_j = static_cast<int>(number_circulant_bins);
			index_k_plus_j
					< static_cast<int>(number_circulant_bins
							+ number_non_circulant_areas); index_k_plus_j++) {
		_vector_array[index_k_plus_j] = _vector_array[index_k_plus_j
				- number_circulant_bins]
				- number_circulant_bins * factor * emintau;
		factor *= tau / ++J;
	}

	std::transform(_vector_array.begin(),
			_vector_array.begin() + number_circulant_bins
					+ number_non_circulant_areas + 1, // one past the array, guaranteed to exist by assert
			_vector_array.begin(),
			std::bind2nd(std::divides<double>(), number_circulant_bins));

	return true;
}

bool VArray::FillArrayWithGarbage() {
	std::fill(_vector_array.begin(), _vector_array.end(), -999);

	return true;
}
/*
 double VArray::V
 (
 Index index_circulant,
 Index index_non_circulant
 ) const
 {
 assert( index_circulant     < _number_of_circulant_bins);
 assert( index_non_circulant < _number_of_non_circulant_areas);
 return _vector_array[index_circulant + index_non_circulant];
 }
 */
bool VArray::CheckInNumbers(Number number_of_circulant_bins,
		Number number_of_non_circulant_areas) {
	_number_of_circulant_bins = number_of_circulant_bins;
	_number_of_non_circulant_areas = number_of_non_circulant_areas;

	return true;
}
} /* namespace populist */
} /* namespace MPILib */
