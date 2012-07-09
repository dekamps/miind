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

#ifndef LAYERMAPPINGLIB_FEATUREMAPITERATORCODE_H
#define LAYERMAPPINGLIB_FEATUREMAPITERATORCODE_H

#include "FeatureMapIterator.h"

using namespace LayerMappingLib;

template<class T>
inline FeatureMapIterator<T>::FeatureMapIterator( T* position,
		int x_skip,
		int y_skip,
		int row_stride,
		int _padding_width) :
	_p_position( position ),
	_p_row_start( position ),
	_x_skip( x_skip ),
	_y_skip( y_skip ),
	_row_stride( row_stride ), 
	_padding_width( _padding_width )
{
}

template<class T>
inline FeatureMapIterator<T>::FeatureMapIterator( const FeatureMapIterator<T>& i ) :
	_p_position( i._p_position ),
	_p_row_start( i._p_position ),
	_x_skip( i._x_skip ),
	_y_skip( i._y_skip ),
	_row_stride( i._row_stride ),
	_padding_width( i._padding_width )
{
}

template<class T>
inline typename FeatureMapIterator<T>::value_type& FeatureMapIterator<T>::operator*()
{
	return *_p_position;
}

template<class T>
inline FeatureMapIterator<T>& FeatureMapIterator<T>::operator++()
{
	_p_position += _x_skip;

	if( _p_position - _p_row_start >= ( _row_stride - _padding_width ) )
	{
		_p_row_start += _row_stride * _y_skip;
		_p_position = _p_row_start;
	}

	return *this;
}

template<class T>
inline FeatureMapIterator<T> FeatureMapIterator<T>::operator++( int )
{
	FeatureMapIterator<T> r = *this;
	++*this;
	return r;
}

template<class T>
inline bool FeatureMapIterator<T>::operator!=( const FeatureMapIterator<T>& x )
{
	return !( _p_position == x._p_position &&
		 _p_row_start == x._p_row_start );
}

template<class T>
inline bool FeatureMapIterator<T>::operator==( const FeatureMapIterator<T>& x )
{
	return ( _p_position == x._p_position &&
		 _p_row_start == x._p_row_start );
}

#endif //LAYERMAPPINGLIB_FEATUREMAPITERATORCODE_H
