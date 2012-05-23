// Copyright (c) 2005 - 2007 Marc de Kamps, Johannes Drever, Melanie Dietz
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

#ifndef LAYERMAPPINGLIB_RECEPTIVEFIELDITERATORCODE_H
#define LAYERMAPPINGLIB_RECEPTIVEFIELDITERATORCODE_H

#include "ReceptiveFieldIterator.h"

using namespace LayerMappingLib;

template<class T>
inline ReceptiveFieldIterator<T>::ReceptiveFieldIterator( T* p, int rf_width, int y_inc ) :
		_p_position( p ),
		_p_row_start( p ),
		_rf_width( rf_width ),
		_y_inc( y_inc )
{
}

template<class T>
inline ReceptiveFieldIterator<T>::ReceptiveFieldIterator() :
		_p_position( NULL ),
		_p_row_start( NULL ),
		_rf_width( 0 ),
		_y_inc( 0 )
{
}

template<class T>
inline typename ReceptiveFieldIterator<T>::value_type& ReceptiveFieldIterator<T>::operator*()
{
	return *_p_position;
}

template<class T>
inline ReceptiveFieldIterator<T>& ReceptiveFieldIterator<T>::operator++()
{
	_p_position++;
	if( _p_position - _p_row_start >= _rf_width )
	{
		_p_position = _p_row_start + _y_inc;
		_p_row_start = _p_position;
	}
	return *this;
}

template<class T>
inline ReceptiveFieldIterator<T> ReceptiveFieldIterator<T>::operator++( int )
{
	ReceptiveFieldIterator<T> r = *this;
	++*this;
	return r;
}

template<class T>
inline ReceptiveFieldIterator<T> ReceptiveFieldIterator<T>::operator+( int x )
{
	ReceptiveFieldIterator<T> r( *this );

	int x_offset = x % _rf_width;
	int y_offset = x / _rf_width;

	r._p_row_start = _p_row_start + ( y_offset * _y_inc );
	r._p_position = _p_position + x_offset + ( y_offset * _y_inc );
	
	return r;
}

template<class T>
inline ReceptiveFieldIterator<T>& ReceptiveFieldIterator<T>::operator+=( int x )
{
	*this = ( *this + x );
	return *this;
}

template<class T>
inline bool ReceptiveFieldIterator<T>::operator!=( const ReceptiveFieldIterator<T>& i )
{
	return !( _p_position == i._p_position &&
		_p_row_start == i._p_row_start );
}

template<class T>
inline bool ReceptiveFieldIterator<T>::operator==( const ReceptiveFieldIterator<T>& i )
{
	return ( _p_position == i._p_position &&
		_p_row_start == i._p_row_start );
}

template<class T>
inline typename ReceptiveFieldIterator<T>::difference_type ReceptiveFieldIterator<T>::operator-( const ReceptiveFieldIterator<T>& i )
{
	assert( _rf_width == i._rf_width );
	assert( _y_inc == i._y_inc );

	T* ip = i._p_position;
	T* tp = _p_position;

	if( ip > tp )
	{
		swap( ip, tp );
	}
	
	difference_type id = tp - ip;
	difference_type xoff = ( id % _y_inc );

	return ( ( ( id - xoff )  / _y_inc ) * _rf_width ) + xoff;
}

#endif //LAYERMAPPINGLIB_RECEPTIVEFIELDITERATORCODE_H
