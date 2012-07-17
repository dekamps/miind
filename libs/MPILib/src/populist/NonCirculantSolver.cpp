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
#include <cassert>
#include <algorithm>
#include <MPILib/include/populist/NonCirculantSolver.hpp>

namespace MPILib {
namespace populist {

NonCirculantSolver::NonCirculantSolver(CirculantMode mode):
AbstractNonCirculantSolver(mode)
{
}

void NonCirculantSolver::ExecuteIntegerExcitatory
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

		for (int i_n_c = 1; i_n_c <= area_n_c; i_n_c++ ){

			int i_stride = i_first + i_n_c*H;
			int i_factor = area_n_c - i_n_c;
			
			assert( i_stride >= 0 && i_stride < static_cast<int>(n_bins) );
			assert( i_factor >= 0 && i_factor < static_cast<int>(n_non_circulant) );

			sum += array_state[i_stride]*_array_factor[i_factor];
		}
  
		array_state[i_bin] = sum;
	}
}

void NonCirculantSolver::ExecuteIntegerInhibitory
(
	Number n_bins,
	Time tau
)
{

	std::valarray<double>& array_state = *_p_array_state;
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
}

void NonCirculantSolver::ExecuteExcitatory
(
	Number n_bins,
	Time   tau
)
{
	assert(n_bins <= _p_array_state->size() );
	if (_mode == INTEGER )
		this->ExecuteIntegerExcitatory(n_bins,tau);
	else
		this->ExecuteFractionExcitatory(n_bins,tau);

}

void NonCirculantSolver::ExecuteFractionExcitatory
(
	Number n_bins,
	Time tau
)
{
	std::valarray<double>& array_state = *_p_array_state;
	double h = _p_input_set->_H_exc +_p_input_set->_alpha_exc;
	if (h == 0 )
		return;

	InitializeArrayFactor(tau, static_cast<int>(n_bins/h)+1); 
	Number n_term_max = this->NumberFactor(); //array_factor is only calculated up to this number of terms

	for (int i = n_bins -1; i >= 0; i--)
	{
		array_state[i] *= _array_factor[0];

		// in the loop below n_walk_back is the maximum term of array_rho that will be accessed.
		// this means is must be curtailed to n_term_max-1, to prevent access to elements of array_factor 
		// that were not calculated because they were to small
		Number n_walk_back = std::min(static_cast<Number>(floor(i/h)),n_term_max-1);
		double step_back = 0;

		for (Index j = 1; j <= n_walk_back; j++){
			step_back += h;
			int lower = i - static_cast<int>(step_back);
			int higher = lower - 1;

			double alpha_high = step_back - floor(step_back);
			double alpha_low  = 1 - alpha_high;
		
			array_state[i] += (alpha_low*array_state[lower] + alpha_high*array_state[higher])*_array_factor[j];
		}

		// if condition below is true, probability density is shared between bin 0 and bin -1.
		// bin -1 must be discarded, but bin 0 still counts, however, again this only makes
		// sense if the terms in array_factor have been calculated
		if (floor((i+1)/h) > floor(i/h) ){
			Number n_walk_to_zero = n_walk_back + 1;
			if (n_walk_to_zero < n_term_max){ 
				step_back += h;
				int lower = i - static_cast<int>(step_back);
				if (lower == 0 ){
					double alpha_low = 1 - (step_back - floor(step_back));
					array_state[i] += (alpha_low*array_state[lower])*_array_factor[n_walk_to_zero];
				}
			}
		}

	}
}


void NonCirculantSolver::ExecuteFractionInhibitory
(
	Number n_bins,
	Time tau
)
{
	std::valarray<double>& array_state = *_p_array_state;
	double h = _p_input_set->_H_inh +_p_input_set->_alpha_inh;
	if (h == 0 )
		return;

	InitializeArrayFactor(tau, static_cast<int>(n_bins/h)+1); 
	Number n_term_max = this->NumberFactor(); //array_factor is only calculated up to this number of terms
	for (Index i = 0; i <= n_bins -1; i++)
	{
		array_state[i] *= _array_factor[0];

		// in the loop below n_walk_back is the maximum term of array_rho that will be accessed.
		// this means is must be curtailed to n_term_max-1, to prevent access to elements of array_factor 
		// that were not calculated because they were to small
	
		Number n_walk_forward = std::min(static_cast<Number>(floor((n_bins- 1 - i)/h)),n_term_max-1);
		double step_forward = 0;

		for (Index j = 1; j <= n_walk_forward; j++){
			step_forward += h;
			int lower = i + static_cast<int>(step_forward);
			int higher = lower + 1;

			double alpha_high = step_forward - floor(step_forward);
			double alpha_low  = 1 - alpha_high;
		
			array_state[i] += (alpha_low*array_state[lower] + alpha_high*array_state[higher])*_array_factor[j];
		}

		// if condition below is true, probability density is shared between bin n-1  and bin n (which does not exist ..).
		// bin n must be discarded, but bin n-1 still counts, however, again this only makes
		// sense if the terms in array_factor have been calculated
		if (floor((i+1)/h) > floor(i/h) ){
			Number n_walk_to_zero = n_walk_forward + 1;
			if (n_walk_to_zero < n_term_max){ 
				step_forward += h;
				Index lower = i + static_cast<Index>(step_forward);
				if (lower == n_bins-1 ){
					double alpha_low = 1 - (step_forward - floor(step_forward));
					array_state[i] += (alpha_low*array_state[lower])*_array_factor[n_walk_to_zero];
				}
			}
		}

	}
}

void NonCirculantSolver::ExecuteInhibitory
(
	Number n_bins,
	Time   tau
)
{
	assert(n_bins <= _p_array_state->size() );
	if (_mode == INTEGER)
		this->ExecuteIntegerInhibitory(n_bins,tau);
	else
		this->ExecuteFractionInhibitory(n_bins,tau);

}


double NonCirculantSolver::Integrate(Number number_of_bins) const
{
	// This routine sums the probability density in each bin
	double f_return = 0;
	for 
	( 
		int index = 0; 
		index <  static_cast<int>(number_of_bins); 
		index++ 
	)
		f_return += (*_p_array_state)[index];

	return f_return;

}

NonCirculantSolver* NonCirculantSolver::Clone() const
{
	return new NonCirculantSolver(*this);
}

} /* namespace populist */
} /* namespace MPILib */
