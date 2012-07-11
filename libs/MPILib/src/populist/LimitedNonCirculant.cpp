// Copyright (c) 2005 - 2009 Marc de Kamps
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
#include <MPILib/include/populist/LimitedNonCirculant.hpp>
#include <MPILib/include/populist/AbstractCirculantSolver.hpp>

namespace MPILib {
namespace populist {

LimitedNonCirculant::LimitedNonCirculant():
AbstractNonCirculantSolver(INTEGER)
{
}

LimitedNonCirculant::~LimitedNonCirculant()
{
}

LimitedNonCirculant* LimitedNonCirculant::Clone() const
{
	return new LimitedNonCirculant(*this);
}

void LimitedNonCirculant::ExecuteExcitatory
(
	Number n_bins,
	Time   tau
)
{

	std::valarray<double>& array_state = *_p_array_state;

	int H = _p_input_set->_H_exc;
	int n_non_circulant = _p_input_set->_n_noncirc_exc;

	InitializeArrayFactor(tau, n_non_circulant); 


	int i_highest = n_bins - 1;
	int i_0 = i_highest%H;
	for ( int i_bin = i_highest; i_bin >= 0; i_bin--){

		int area_n_c  = (i_bin + H - i_0 - 1)/H;
		int i_first = i_bin - area_n_c*H;

		assert(area_n_c >= 0 && area_n_c < n_non_circulant);
		assert(i_first < static_cast<int>(n_bins) );

		double sum = (i_first >= 0) ? _array_factor[area_n_c]*array_state[i_first] : 0; 

		int i_lower = std::max(1, area_n_c - static_cast<int>(NONCIRC_LIMIT) );
		for (int i_n_c = i_lower; i_n_c <= area_n_c; i_n_c++ ){

			int i_stride = i_first + i_n_c*H;
			int i_factor = area_n_c - i_n_c;
			
			assert( i_stride >= 0 && i_stride < static_cast<int>(n_bins) );
			assert( i_factor >= 0 && i_factor < static_cast<int>(n_non_circulant) );

			sum += array_state[i_stride]*_array_factor[i_factor];
		}
  
		array_state[i_bin] = sum;
	}
}


void LimitedNonCirculant::ExecuteInhibitory
(
	Number n_bins,
	Time   tau
)
{
/*	valarray<double>& array_state = *_p_array_state;
	int H = _p_input_set->_H_inh;
	int n_non_circulant = _p_input_set->_n_noncirc_inh;

	InitializeArrayFactor(tau, n_non_circulant); 


	int i_highest = n_bins - 1;
	int i_0 = i_highest%H;
	for ( int i_bin = 0; i_bin < static_cast<int>(n_bins); i_bin++ ){

		// set i_bin -> n - 1 -i_bin in excitatory and you get this:
		int area_n_c  = (n_bins - i_bin + H - i_0 - 2)/H;
		int i_first = i_bin + area_n_c*H;

		// i_factor runs down where n_c runs up, just as in excitatory
		double sum = (i_first < static_cast<int>(n_bins)) ? _array_factor[area_n_c]*array_state[i_first] : 0; 

		for (int i_n_c = 1; i_n_c <= area_n_c; i_n_c++ ){

			// but the stride counts down
			int i_stride = i_first - i_n_c*H;
			int i_factor = area_n_c - i_n_c;

			sum += array_state[i_stride]*_array_factor[i_factor];
		}
  
		array_state[i_bin] = sum;
	}
*/
}

void LimitedNonCirculant::InitializeArrayFactor
(
	Time   tau,
	Number n_non_circulant
)
{
	if ( n_non_circulant > _array_factor.size() )
		_array_factor.resize(n_non_circulant);

	_array_factor[0] = exp(-tau);
	for (Index i = 1; i < NONCIRC_LIMIT; i++)
		_array_factor[i] = tau*_array_factor[i - 1]/i;
}
} /* namespace populist */
} /* namespace MPILib */
