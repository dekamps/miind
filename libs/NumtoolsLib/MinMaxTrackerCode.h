// Copyright (c) 2005 - 2009 Marc de Kamps, Korbinian Trumpp
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
// Coded by Korbo...
#ifndef _CODE_LIBS_NUMTOOLSLIB_MINMAXTRACKERCODE_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_MINMAXTRACKERCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#undef min
#undef max
#endif

#include "MinMaxTracker.h"
#include <iostream>

using namespace std;

namespace NumtoolsLib
{
	template<class ValueType>
	MinMaxTracker<ValueType>::MinMaxTracker() :
	_min( (_limits.has_infinity) ? _limits.infinity() : _limits.max() ),
		_max( (_limits.has_infinity) ? -_limits.infinity() : _limits.min() ),
		_sum_for_avg( 0 ),
		_nr_of_values( 0 ),
		_min_index( 0 ),
		_max_index( 0 )
	{
	}

	//------------------------------------------------------

	template<class ValueType>
	void MinMaxTracker<ValueType>::reset()
	{
		_min = (_limits.has_infinity) ? _limits.infinity() : _limits.max();
		_max = (_limits.has_infinity) ? -_limits.infinity() : _limits.min();
		_sum_for_avg = 0;
		_nr_of_values = 0;
		_min_index = 100000;
		_max_index = 100000;
	}
	
	//------------------------------------------------------

	template<class ValueType>
	void MinMaxTracker<ValueType>::feedValue( ValueType value )
	{
		_nr_of_values++;
		if( value > _max )
			_max = value;
		if( value < _min )
			_min = value;

		_sum_for_avg += value;
	}
	
	//------------------------------------------------------

	template<class ValueType>
	void MinMaxTracker<ValueType>::feedValue( ValueType value, unsigned int index )
	{
		_nr_of_values++;
		if( value > _max )
		{
			_max_index = index;
			_max = value;
		}
		if( value < _min )
		{
			_min_index = index;
			_min = value;
		}
		_sum_for_avg += value;
	}
	
	//------------------------------------------------------

	template<class ValueType>
	ValueType MinMaxTracker<ValueType>::getMin()
	{
		return _min;
	}
	
	//------------------------------------------------------

	template<class ValueType>
	ValueType MinMaxTracker<ValueType>::getMax()
	{
		return _max;
	}
	
	//------------------------------------------------------

	template<class ValueType>
	unsigned int MinMaxTracker<ValueType>::getMinIndex()
	{
		return _min_index;
	}
	
	//------------------------------------------------------

	template<class ValueType>
	unsigned int MinMaxTracker<ValueType>::getMaxIndex()
	{
		return _max_index;
	}
	
	//------------------------------------------------------

	template<class ValueType>
	ValueType MinMaxTracker<ValueType>::getAvg()
	{
		return _sum_for_avg / _nr_of_values;
	}
	
	//------------------------------------------------------

	template<class ValueType>
	int MinMaxTracker<ValueType>::nrOfFeeds()
	{
		return _nr_of_values;
	}
	
} // end of Numtools


#endif // include guard
