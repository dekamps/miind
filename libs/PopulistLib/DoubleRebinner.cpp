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
#include "AbstractZeroLeakEquations.h"
#include "DoubleRebinner.h"
#include <iostream>

using namespace std;
using namespace PopulistLib;

DoubleRebinner::DoubleRebinner():
_p_array_state           (0),
_number_original_growing (0),
_i_reversal              (0),
_i_odd                   (0)
{
}

DoubleRebinner::~DoubleRebinner()
{
}

bool DoubleRebinner::Configure
(
	valarray<double>&	array_state,
	Index			index_reversal_bin,
	Index			index_reset_bin,
	Number			number_expanded, // this number is used as a check
	Number			number_original 
)
{

	_i_reversal    = static_cast<int>(index_reversal_bin);
	_i_odd     = _i_reversal % 2;

	_number_original_growing = number_original - _i_reversal;
	_number_expanded_growing = 2*_number_original_growing;


	if ( 2*_number_original_growing != _number_expanded_growing)
		return false;
	_p_array_state = &array_state;

	return true;
}

void DoubleRebinner::RebinPositive()
{
	valarray<double>& array_state = *_p_array_state;
	
	for (int i = 0; i < _number_original_growing; i++)
	{
		array_state[i + _i_reversal] = array_state[2*i + _i_reversal] + array_state[2*i + _i_reversal + 1];
//		cout << i + _i_reversal << " " <<  array_state[i + _i_reversal] << 2*i + _i_reversal << " " << 2*i + _i_reversal + 1 << " " << array_state[2*i + _i_reversal + 1] << endl;
	}



	for (int i_pos = _number_original_growing + _i_reversal; i_pos < static_cast<int>(array_state.size()); i_pos++)
		array_state[i_pos] = 0;
	

}

void DoubleRebinner::RebinNegative()
{
	valarray<double>& array_state = *_p_array_state;
	int n_negative = _i_reversal/2;
		
	for (int j = 1; j <= n_negative; j++)	
		array_state[_i_reversal - j] = array_state[_i_reversal - 2*j] + array_state[_i_reversal -2*j + 1]/2;
	

	double rest = array_state[0];

	int i_rest_start = _i_reversal - n_negative - 1;
	for (int i_rest = i_rest_start; i_rest >= 0; i_rest--)
		array_state[i_rest] = 0;
	if ( _i_odd == 1 && i_rest_start >= 0)
		array_state[i_rest_start] += rest;
	
	cout << i_rest_start << endl;

}


bool DoubleRebinner::Rebin(AbstractZeroLeakEquations*)
{

	RebinPositive();
	RebinNegative();

	return true;
}

double DoubleRebinner::ExpansionFactor() const
{
	return static_cast<double>(_i_reversal + 1 + _number_expanded_growing)/static_cast<double>(_i_reversal + 1 + _number_original_growing);
}

DoubleRebinner* DoubleRebinner::Clone() const
{
	return new DoubleRebinner;
}
