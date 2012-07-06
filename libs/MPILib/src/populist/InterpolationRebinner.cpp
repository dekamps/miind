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
#include <MPILib/include/populist/InterpolationRebinner.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <cassert>
#include <limits>
#include <iostream>
namespace MPILib {
namespace populist {

InterpolationRebinner::InterpolationRebinner():
_p_accelerator(gsl_interp_accel_alloc () ),
_x_array				(0),
_y_array				(0),
_sum_before				(0),
_dv_before				(0),
_dv_after				(0),
_index_reversal_bin		(0),
_index_reset_bin		(0),
_number_original_bins	(0),
_number_new_bins		(0),
_p_spline				(0)
{
}

InterpolationRebinner::~InterpolationRebinner()
{
     //freeing 0 ptr leads to crash in gsl 1.4
     if ( _p_spline )
	     gsl_spline_free(_p_spline );

     gsl_interp_accel_free (_p_accelerator);
}

bool InterpolationRebinner::Rebin
(
	AbstractZeroLeakEquations* p_zl
)
{
	
	_sum_before = _p_array->sum();

	SmoothResetBin			();
	PrepareLocalCopies		();
	Interpolate				();
	ResetOvershoot			();		// set the shrunk negative part of probability to zero
	RescaleAllProbability	(p_zl); // because the scale factor in PopulationGridController will be reset to 1
	ReplaceResetBin			(p_zl); // take reafactive probability into account
	return true;
}

void InterpolationRebinner::SmoothResetBin
(
)
{
	// Purpose: the reset bin usually contains a spike. It is simply being replaced with
	// the average density of its neighbours. The missing density will be reapplied after
	// the rebinning
	// Author: Marc de Kamps
	// Date: 08-01-2006

	std::valarray<double>& array = *_p_array;

	if (_index_reset_bin == 0)
		array[_index_reset_bin] = array[_index_reset_bin + 1];
	else
		if (_index_reset_bin == static_cast<int>(_number_original_bins) - 1)
			// quaint !!
			array[_index_reset_bin] = array[_index_reset_bin - 1];
		else
			array[_index_reset_bin] = (array[_index_reset_bin - 1] + array[_index_reset_bin + 1])/2;
}

bool InterpolationRebinner::Configure
(
	valarray<double>& array,
	Index             index_reversal_bin,
	Index             index_reset_bin,
	Number            number_original_bins,
	Number            number_new_bins
)
{
	assert( number_new_bins - 1 > index_reversal_bin   );
	assert( number_new_bins     <= number_original_bins );

	if ( number_new_bins == number_original_bins )
		return true;

	_index_reversal_bin   = static_cast<int>(index_reversal_bin);
	_index_reset_bin      = static_cast<int>(index_reset_bin);
	_number_original_bins = number_original_bins;
	_number_new_bins      = number_new_bins;

	_p_array              = &array;

	if (_p_spline)
		gsl_spline_free (_p_spline);

	_p_spline = 
		gsl_spline_alloc 
		(
			gsl_interp_cspline, 
			_number_original_bins
		);

	return true;
}

void InterpolationRebinner::PrepareLocalCopies
(
)
{
	valarray<double>& array = *_p_array;

	if ( _number_original_bins > _x_array.size() ){
		_x_array.resize(_number_original_bins);
		_y_array.resize(_number_original_bins);
	}

	std::copy
	(
		&array[0], 
		&array[0] + _number_original_bins,
		_y_array.begin()
	);

	// The -1 is necessary! _number_bins - 1 = index_theta 
	_dv_before = 1.0/static_cast<double>(_number_original_bins - 1 - _index_reversal_bin);
	_dv_after  = 1.0/ static_cast<double>(_number_new_bins     - 1 - _index_reversal_bin);

	for ( int i = 0; i < static_cast<int>(_number_original_bins); i++ )
		_x_array[i] = (i - _index_reversal_bin)*_dv_before;
}

int InterpolationRebinner::IndexNewResetBin()
{
	// Purpose: Locate new area of the new reset bin. If V_reset != V_reversal, they are
	// different.
	// Assumption: dv_before and dv_after have been calculated
	// Author: Marc de Kamps
	// Date: 08-01-2006

	Potential v_reset = (_index_reset_bin - _index_reversal_bin)*_dv_before;
	int new_reset_interval =  static_cast<int>(floor(v_reset/_dv_before+0.5));
	return _index_reversal_bin + new_reset_interval;
}

void InterpolationRebinner::ResetOvershoot()
{
	valarray<double>& array = *_p_array;
	for
	( 
		int index_rest_bins = static_cast<int>(_number_new_bins);
		index_rest_bins < static_cast<int>(_number_original_bins);
		index_rest_bins++
	)
		array[index_rest_bins] = 0;
}

void InterpolationRebinner::Interpolate()
{
	valarray<double>& array = *_p_array;
	gsl_spline_init 
		(
			_p_spline, 
			&_x_array[0], 
			&_y_array[0],
			_number_original_bins
		);
	for 
	(
		int index_new_potential = 0;
		index_new_potential < static_cast<int>(_number_new_bins);
		index_new_potential++
	)
	{

		double x = ( index_new_potential - _index_reversal_bin )*_dv_after;   

		// take care, rebinning enlarges the negative region, for which there was no info available, therefore:
		array[index_new_potential] = (x > _x_array[0]) ?  
					gsl_spline_eval 
					(
						_p_spline, 
						x, 
						_p_accelerator
					) : 0;
	}
}


void InterpolationRebinner::ReplaceResetBin(AbstractZeroLeakEquations* p_zl)
{
	double refractive = p_zl ? p_zl->RefractiveProbability() : 0.0;
	valarray<double>& array = *_p_array;
	array[this->IndexNewResetBin()] += 1.0 - _p_array->sum() - refractive;
}

InterpolationRebinner* InterpolationRebinner::Clone() const
{
	// every useful bit of initialization must happen at Configure
	// this is just type transfer

	return new InterpolationRebinner;
}

void InterpolationRebinner::RescaleAllProbability(AbstractZeroLeakEquations* p_zl){
	double scale = _dv_after/_dv_before; 
	*_p_array *=  scale;
	// do not scale refractive probability here
}
} /* namespace populist */
} /* namespace MPILib */

