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
#include <cassert>
#include <iostream>
#include "AbstractZeroLeakEquations.h"
#include "SinglePeakRebinner.h"

using namespace std;
using namespace PopulistLib;

SinglePeakRebinner::SinglePeakRebinner()
{
}

SinglePeakRebinner::~SinglePeakRebinner()
{
}

bool SinglePeakRebinner::Configure
(
	valarray<double>& array,
	Index index_reversal_bin,
	Index index_reset_bin,
	Number number_original_bins,
	Number number_new_bins
)
{
	assert( number_original_bins > number_new_bins );
	assert( index_reversal_bin   < number_new_bins );

	_p_array              = &array;
	_index_reversal_bin   = index_reversal_bin;
	_index_reset_bin      = index_reset_bin;
	_number_original_bins = number_original_bins;
	_number_new_bins      = number_new_bins;

	return true;
}

bool SinglePeakRebinner::Rebin
(
	AbstractZeroLeakEquations*
)
{
	valarray<double>& array = *_p_array;
	// find the peak, there should be one

	Index index_peak = 0;
	while ( array[index_peak++] == 0 && index_peak < array.size() )
		;

	// there was no peak
	if ( index_peak == array.size() )
		return false;

	--index_peak;

	double current_scale = (static_cast<double>(_number_original_bins)- static_cast<double>(_index_reversal_bin))/\
		                     (static_cast<double>(_number_new_bins) - static_cast<double>(_index_reversal_bin));

	Index new_index   = static_cast<Index>(index_peak/current_scale)  + _index_reversal_bin;

	if ( new_index < index_peak)
	{
		array[new_index] = array[index_peak];
		array[index_peak]= 0;
		return true;
	}
	else
		if (new_index == index_peak)
			return true;
		else
			return false;
}

SinglePeakRebinner* SinglePeakRebinner::Clone() const
{
	// this is just type transfer, any serious initialization happens at Configure
	return new SinglePeakRebinner;
}
