// Copyright (c) 2005 - 2008 Marc de Kamps
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

#include "AlgorithmGrid.h"
#include "DynamicLibException.h"
#include "LocalDefinitions.h"
#include <cassert>
#include <functional>
#include "../UtilLib/UtilLib.h"

using namespace DynamicLib;
using namespace std;
using namespace UtilLib;

AlgorithmGrid::AlgorithmGrid(Number number_of_elements):
_number_state         (number_of_elements),
_array_state          (0.0,number_of_elements),
_array_interpretation (0.0,number_of_elements)
{
}

AlgorithmGrid::AlgorithmGrid(const AlgorithmGrid& rhs):
_number_state         (rhs._number_state),
_array_state          (rhs._array_state),
_array_interpretation (rhs._array_interpretation)
{
	assert( _array_state.size() == _array_interpretation.size() );
}

AlgorithmGrid::AlgorithmGrid
(
	const vector<double>& array_state
):
_number_state(static_cast<Number>(array_state.size() ) ),
_array_state
(
	ToValarray<double>(array_state)
),
_array_interpretation
(
	valarray<double>(0.0,array_state.size())
)
{
}

AlgorithmGrid::AlgorithmGrid
(
	const vector<double>& array_state,
	const vector<double>& array_interpretation
):
_number_state(static_cast<Number>( array_state.size() ) ),
_array_state          
(
	ToValarray<double>(array_state)
),
_array_interpretation
(
	ToValarray<double>(array_interpretation)
)
{
	assert( _array_state.size() == _array_interpretation.size() );
}

AlgorithmGrid& AlgorithmGrid::operator=(const AlgorithmGrid& rhs)
{
	if ( &rhs == this )
		return *this;
	else
	{
		// resize, because copying valarrays of different length is undefined
		_array_state.resize(rhs._array_state.size());
		_array_interpretation.resize(rhs._array_interpretation.size());

		_array_state          = rhs._array_state;
		_array_interpretation = rhs._array_interpretation;
		_number_state         = rhs._number_state;
		return *this;
	}
}

vector<double> AlgorithmGrid::ToStateVector() const
{
	return ToVector
			(
				_array_state, 
				_number_state
			);
}

void AlgorithmGrid::Resize(Number number_of_new_bins) 
{
	_array_state.resize(number_of_new_bins);
	_array_interpretation.resize(number_of_new_bins);
}

vector<double> AlgorithmGrid::ToInterpretationVector() const
{
	return ToVector
			(
				_array_interpretation, 
				_number_state
			);
}

string AlgorithmGrid::Tag() const
{
	return STR_GRID_TAG;
}

bool AlgorithmGrid::ToStream(ostream& s) const
{
	s << Tag() << "\n";

	s << _number_state	<< "\n";

	copy
	(	
		this->begin_state(),
		this->end_state(),
		ostream_iterator<double>(s," ")
	);

	s << "\n" << _number_state << "\n";

	copy 
	(
		this->begin_interpretation(),
		this->end_interpretation(),
		ostream_iterator<double>(s," ")
	);

	s << "\n" << ToEndTag(Tag())    << "\n";

	return true;
}

bool AlgorithmGrid::FromStream(istream &s)
{
	string tag;
	s >> tag;
	if (tag != Tag())
		throw DynamicLibException(STR_GRID_PARSE_ERROR);

	int size_grid;
	s >> size_grid;

	

	bool b_return =  StreamToArray
			(
				s,
				size_grid,
				&_array_state[0]
			);
	if ( ! b_return )
		return false;
	else
	{
		s >> size_grid;

		b_return = StreamToArray
					(
						s,
						size_grid,
						&_array_interpretation[0]
					);

		string str_end_tag;

		s >> str_end_tag;
		if ( str_end_tag != ToEndTag(Tag()) )
			b_return = false;

		return b_return;
	}

}

const double* AlgorithmGrid::begin_state() const
{

	valarray<double>& ref_state = const_cast<valarray<double>& >(_array_state);
	const double* p_begin = &ref_state[0];
	return p_begin;
}

const double* AlgorithmGrid::end_state() const
{
	valarray<double>& ref_state = const_cast<valarray<double>& >(_array_state);
	const double* p_end = &ref_state[_number_state];
	return p_end;
}

const double* AlgorithmGrid::begin_interpretation() const
{

	valarray<double>& ref_state = const_cast<valarray<double>& >(_array_interpretation);
	const double* p_begin = &ref_state[0];
	return p_begin;
}

const double* AlgorithmGrid::end_interpretation() const
{
	valarray<double>& ref_state = const_cast<valarray<double>& >(_array_interpretation);
	const double* p_end = &ref_state[_number_state];
	return p_end;
}

valarray<double>& AlgorithmGrid::ArrayState()
{
	return _array_state;
}

valarray<double>& AlgorithmGrid::ArrayInterpretation()
{
	return _array_interpretation;
}

Number& AlgorithmGrid::StateSize()
{
	return _number_state;
}

Number AlgorithmGrid::StateSize() const
{
	return _number_state;
}