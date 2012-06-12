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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "OrientedPattern.h"
#include "LocalDefinitions.h"



namespace StructnetLib
{

//	template <class PatternValue>
//	stream& operator<<(ostream&, const OrientedPattern<PatternValue>&);

	template <class PatternValue>
	Number OrientedPattern<PatternValue>::NumberOfElements
				(
					Index n_x, 
					Index n_y, 
					Index n_or
				) const
	{
		// total number of pattern elements in this OrientedPattern
		// Tested: 
		// Author: Marc de Kamps

		return n_x*n_y*n_or;
	}


	template <class PatternValue>
	Number OrientedPattern<PatternValue>::NXFromIndex(Index nr_index) const
	{
		// which x-coordinate does this index refer to ?
		// Tested: 26-01-2000
		// Author: Marc de Kamps

		size_t nr_offset = nr_index - this->n_or_from_index(nr_index)*_nr_x*_nr_y;

		return nr_offset%_nr_y;
	}


	template <class PatternValue>
	Number OrientedPattern<PatternValue>::NYFromIndex( Index nr_index ) const
	{

		// which y-coordinate does this index refer to ?
		// Tested: 26-01-2000
		// Author: Marc de Kamps
		
		size_t nr_offset = nr_index - this->n_or_from_index(nr_index)*_nr_x*_nr_y;
		return nr_offset/_nr_x;
	}


	template <class PatternValue>
	Number OrientedPattern<PatternValue>::NORFromIndex( Index nr_index ) const
	{
		// which y-coordinate does this index refer to ?
		// Tested: 26-01-2000
		// Author: Marc de Kamps
	
		return  nr_index/(_nr_x*_nr_y);
	}

	
	template <class PatternValue>
	Index OrientedPattern<PatternValue>::IndexFunction(Index n_x, Index n_y, Index nr_or ) const
	{
		// Index is the inverse of the previous functions. Given x,y-coordinates, 
		// orientation, it produces the array index of the element
		// Tested: 
		// Author: Marc de Kamps

		 
		Index n_starting_element = nr_or*_nr_x*_nr_y;

		return n_starting_element + n_y*_nr_y + n_x;
	}



	template <class PatternValue>
	OrientedPattern<PatternValue>::OrientedPattern
	( 
		Number n_x,
		Number n_y,
		Number nr_o 
	):
	Pattern<PatternValue>(NumberOfElements(n_x,n_y,nr_o)),
	_nr_x(n_x),
	_nr_y(n_y),
	_nr_o(nr_o)
	{
	}

	template <class PatternValue>
	OrientedPattern<PatternValue>::OrientedPattern
	(
		const OrientedPattern<PatternValue>& rhs
	):
	Pattern<PatternValue>(rhs),
	_nr_x(rhs._nr_x),
	_nr_y(rhs._nr_y),
	_nr_o(rhs._nr_o)
	{
	}

	template <class PatternValue>
	OrientedPattern<PatternValue>&
		OrientedPattern<PatternValue>::operator=
		(
			const OrientedPattern<PatternValue>& rhs
		)
	{
		if (this == &rhs)
			return *this;

		else 
		{
			Pattern<PatternValue>::operator=(rhs);
			_nr_x = rhs._nr_x;
			_nr_y = rhs._nr_y;
			_nr_o = rhs._nr_o;

			return *this;
		}
	}
	template <class PatternValue>
	OrientedPattern<PatternValue>::OrientedPattern(istream& s):
	Pattern<PatternValue>(0)
	{
		FromStream(s);
	}

	template <class PatternValue>
	bool OrientedPattern<PatternValue>::FromStream(istream& s)
	{
		string str;

		s >> str;

		if ( str != OrientedPattern<PatternValue>::Tag() )
			throw PatternParsingException("Oriented pattern begin tag expected");

		Pattern<PatternValue>::FromStream(s);
		int n_total;
		s >>  n_total; // not used atm
		s >> _nr_x;
		s >> _nr_y;
		s >> _nr_o;

		if ( ! s.good() )
			throw PatternParsingException("int expected");

		s >> str;

		if ( str != this->ToEndTag(OrientedPattern<PatternValue>::Tag()) )
			throw PatternParsingException("Oriented Pattern error");

		return true;
	}

	template <class PatternValue>
	bool OrientedPattern<PatternValue>::ToStream(ostream& s) const
	{
		s << OrientedPattern<PatternValue>::Tag() << "\n";

		if ( ! Pattern<PatternValue>::ToStream(s) )
			return false;

		Number nr_of_elements = NumberOfElements(_nr_x,_nr_y,_nr_o);

		s << nr_of_elements << "\n";
		s << _nr_x          << "\n";
		s << _nr_y          << "\n";
		s << _nr_o          << "\n";

		s << this->ToEndTag(OrientedPattern<PatternValue>::Tag()) << "\n";

		return true;
	}

template <class PatternValue>
ostream& operator<<(ostream& s, const OrientedPattern<PatternValue>& pattern)
{
	pattern.ToStream(s);
	return s;
}		

template <class PatternValue>
PatternValue& OrientedPattern<PatternValue>::operator ()(Index n_x, Index n_y, Index n_or )
{
	return this->operator[]( IndexFunction(n_x,n_y,n_or) );
}											

template <class PatternValue>
const PatternValue& OrientedPattern<PatternValue>::operator ()(Index n_x, Index n_y, Index n_or ) const 
{
	return this->operator[]( IndexFunction(n_x,n_y,n_or) );
}											


template <class PatternValue>
bool OrientedPattern<PatternValue>::MaxTrx(int* tr) const
{
	int maxval = -std::numeric_limits<int>::max();

	for (int n_x = _nr_x - 1; n_x >= 0 ; n_x-- )
		for (int n_y = 0; n_y < _nr_y; n_y++ )
			for(int n_o = 0; n_o < _nr_o; n_o++ )
				if ( operator()(n_x,n_y,n_o) != 0 && n_x > maxval)
					maxval = n_x;

	if ( maxval < std::numeric_limits<int>::max() )
	{
		*tr = _nr_x - maxval - 1;
		return true;
	}
	else
		return false;
}

template <class PatternValue>
bool OrientedPattern<PatternValue>::MinTrx(int* tr) const
{	
	int minval = std::numeric_limits<int>::max();

	for ( int n_x = 0 ; n_x < _nr_x ; n_x++ )
		for ( int n_y = 0; n_y < _nr_y; n_y++ )
			for( int n_or = 0; n_or < _nr_o; n_or++ )
				if ( operator()(n_x,n_y,n_or) != 0 && n_x < minval)
					minval = n_x;
	

	if (minval > -1)
	{
		*tr = -minval;
		return true;
	}
	else 
		return false;
}	

template <class PatternValue>
bool OrientedPattern<PatternValue>::MaxTry(int* tr) const
{
	int maxval = -std::numeric_limits<int>::max();

	for ( int n_y = _nr_y - 1; n_y >= 0 ; n_y-- )
		for ( int n_x = 0; n_x < _nr_x; n_x++ )
			for( int n_or = 0; n_or < _nr_o; n_or++ )
				if ( operator()(n_x,n_y,n_or) != 0 && n_y > maxval )
					maxval = n_y;

	if (maxval < std::numeric_limits<int>::max())
	{
		*tr = _nr_y - maxval - 1;
		return true;
	}
	else
		return false;
}

template <class PatternValue>
bool OrientedPattern<PatternValue>::MinTry(int *tr) const
{	
	int minval = std::numeric_limits<int>::max();

	for (int n_x = 0 ; n_x < _nr_x ; n_x++ )
		for (int n_y = 0; n_y < _nr_y; n_y++ )
			for (int n_or = 0; n_or < _nr_o; n_or++ )
				if ( operator()(n_x,n_y,n_or) != 0 && n_y < minval)
					minval = n_y;
	if ( minval > -1 )
	{
		*tr = -minval;
		return true;
	}
	else
		return false;
}	
	
template <class PatternValue>
bool OrientedPattern<PatternValue>::TransX( int i_x_shift )
{
	int mintrx = 0;
	int maxtrx = 0;
	this->MinTrx(&mintrx);
	this->MaxTrx(&maxtrx);
	if (i_x_shift < 0 && i_x_shift < mintrx)
		return false;
	if (i_x_shift > 0 && i_x_shift > maxtrx)
		return false;
	
	int n_x, n_y, n_o;
	if ( i_x_shift > 0)
	{
		// move from right to left, copying from left to right so as not to
		// overwrite values that still need to be copied

		for ( n_x = _nr_x - 1; n_x >= i_x_shift ; n_x-- )
			for (  n_y = 0; n_y < _nr_y; n_y++ )
				for ( n_o = 0; n_o < _nr_o; n_o++ )
					operator()( n_x , n_y, n_o ) = operator()( n_x - i_x_shift, n_y, n_o );
		// the first i-shift columns must be filled with zeros

		int n_max = std::min<int>(i_x_shift, _nr_x );
		for( n_x = 0; n_x < n_max; n_x++ )
			for ( n_y = 0; n_y < _nr_y; n_y++ )
				for ( n_o = 0; n_o < _nr_o; n_o++ )
					operator()(n_x, n_y, n_o ) = 0;
	}
	
	else
	{
		// move from left to right, copying values from right to left,
		// so as not to overwrite values that still must be copied

		for ( n_x = 0; n_x < _nr_x + i_x_shift; n_x++ )
			for ( n_y = 0; n_y < _nr_y; n_y++ )
				for ( n_o = 0; n_o < _nr_o; n_o++ )
					operator()( n_x, n_y, n_o ) = operator()( n_x - i_x_shift, n_y, n_o );

		int n_min = max( 0, static_cast<int>(_nr_x) - 1 + i_x_shift );
		for ( n_x = _nr_x - 1; n_x > n_min; n_x-- ) 
			for ( n_y = 0; n_y < _nr_y; n_y++ )
				for( n_o = 0; n_o < _nr_o; n_o++ )
					operator()(n_x, n_y, n_o) = 0; 	

	}	
	return true;
}

template <class PatternValue>
string OrientedPattern<PatternValue>::Tag() const
{
	return STR_TAG_ORIENTED_PATTERN;
}

template <class PatternValue>
bool OrientedPattern<PatternValue>::TransY(int i_y_shift )
{			
	int mintry = 0;
	int maxtry = 0;
	this->MinTry(&mintry);
	this->MaxTry(&maxtry);
	if (i_y_shift < 0 && i_y_shift < mintry)
		return false;
	if (i_y_shift > 0 && i_y_shift > maxtry)
		return false;

	int n_x, n_y, n_o;
	if ( i_y_shift > 0)
	{
		// move from bottom to top, copying from top to bottom so as not to
		// overwrite values that still need to be copied
		for ( n_y = _nr_y - 1; n_y >= i_y_shift ; n_y-- )
			for ( n_x = 0; n_x < _nr_x; n_x++ )
				for( n_o = 0; n_o < _nr_o; n_o++)
					operator()( n_x , n_y, n_o ) = operator()( n_x, n_y - i_y_shift, n_o);

		int n_y_max = std::min<int>( i_y_shift, _nr_y );
		for( n_y = 0; n_y < n_y_max; n_y++ )
			for ( n_x = 0; n_x < _nr_x; n_x++ )
				for( n_o = 0; n_o < _nr_o; n_o++ )
					operator()(n_x, n_y, n_o ) = 0;
	}
	
	else
	{
		// move from top to bottom, copying values from bottom to top,
		// so as not to overwrite values that still must be copied
	
		for ( n_y = 0; n_y < _nr_y + i_y_shift; n_y++ )
			for ( n_x = 0; n_x < _nr_x; n_x++ )
				for ( n_o = 0; n_o < _nr_o; n_o++ )
					operator()( n_x, n_y, n_o ) = operator()( n_x, n_y - i_y_shift, n_o );

		int n_y_min = max( 0, _nr_y - 1 + i_y_shift );
		for ( n_y = _nr_y - 1; n_y >= n_y_min; n_y-- ) 
			for ( n_x = 0; n_x < _nr_x; n_x++ )
				for ( n_o = 0; n_o < _nr_o; n_o++ )
					operator()(n_x, n_y, n_o) = 0; 	

	}	
	return true;
}

template <class PatternValue>
Number OrientedPattern<PatternValue>::CountNonZero() const
{
	Number n_pat = _nr_x*_nr_y*_nr_o;
	Index n_ret = 0;

	for (Index n_ind = 0; n_ind < n_pat; n_ind++ )
			if ( this->operator[](n_ind) != 0 )
				 n_ret++;

	return n_ret;
}

template <class PatternValue>
bool operator==(const OrientedPattern<PatternValue>& pat_l, const OrientedPattern<PatternValue>& pat_r)
{
	if (pat_l.Size() != pat_r.Size() )
		return false;

	for (Index i = 0; i < pat_l.Size(); i++ )
		if ( pat_l[i] != pat_r[i] )
			return false;

	return true;
}

} // end of Strucnet

