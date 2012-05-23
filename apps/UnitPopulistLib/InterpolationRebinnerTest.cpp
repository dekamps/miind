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
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>
#include <PopulistLib.h>

BOOST_AUTO_TEST_CASE(InterpolationRebinnerTest){
	// Create a linear density profile which extends from the threshold
	// down to the reversal bin. A big peak is present in the reset bin
	// The rebinned probability must have the same gradient and the same total amount of
	// probability

	const Number n_state   = 100;
	const Number new_state = 80;
	const double alpha = 0.2;
	Index i_reversal = 10;
	Index i_reset    = 15;
	valarray<double> state(n_state);
	for (Index i = n_state -1; i >= i_reversal; i--)
		state[i]=alpha*(n_state-i-1);
	state[i_reset] = 100;

	double gradient_before = state[n_state-1] - state[i_reversal];
	InterpolationRebinner rebin;

	rebin.Configure(state,i_reversal,i_reset,n_state,new_state);
	rebin.Rebin(0);

	double sum_after = state.sum();
	double gradient_after = (state[new_state-1] - state[i_reversal]);

	double corrected_gradient = gradient_after*(new_state - 1 - i_reversal)/(n_state - 1 - i_reversal);
	BOOST_CHECK_CLOSE(1.0,sum_after,1e-10);
	BOOST_CHECK_CLOSE(gradient_before,corrected_gradient,1e-10);

}
