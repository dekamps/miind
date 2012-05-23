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

#ifndef LAYERMAPPINGLIB_FEATUREMAPCODE_H
#define LAYERMAPPINGLIB_FEATUREMAPCODE_H

#include "FeatureMap.h"

using namespace LayerMappingLib;
using namespace std;

template<class T>
FeatureMap<T>::FeatureMap( int width, int height, int padding_width, int padding_height ) :
	_p_start( new T[ ( width + padding_width ) * ( height + padding_height ) ] ),
	_ref_count( new int ),
	_width( width + padding_width ),
	_height( height + padding_height ),
	_padding_width( padding_width ),
	_padding_height( padding_height )
{
	assert( padding_width % 2 == 0 );
	assert( padding_height % 2 == 0 );

	*_ref_count = 1;

	fill( _p_start,
		_p_start + _width * _height,
		0 );
}

template<class T>
FeatureMap<T>::FeatureMap() :
	_p_start( NULL ),
	_ref_count( NULL ),
	_width( -1 ),
	_height( -1 ),
	_padding_width( -1 ),
	_padding_height( -1 )
{
}

template<class T>
FeatureMap<T>::FeatureMap( const FeatureMap<T>& sl ) :
	_p_start( sl._p_start ),
	_ref_count( sl._ref_count ),
	_width( sl._width ),
	_height( sl._height ),
	_padding_width( sl._padding_width ),
	_padding_height( sl._padding_height )
{
	_ref_count_inc();
}

template<class T>
void FeatureMap<T>::_ref_count_inc()
{
	if( _ref_count != NULL )
	{
		(*_ref_count)++;
	}
}

template<class T>
void FeatureMap<T>::_ref_count_dec()
{
	if( _ref_count != NULL )
	{
		(*_ref_count)--;

		if( (*_ref_count) == 0 )
		{
			_deallocate();
		}
	}
}

template<class T>
void FeatureMap<T>::_deallocate()
{
	if( _ref_count != NULL )
	{
		delete _ref_count;
	}
	if( _p_start != NULL )
	{
		delete[] _p_start;
	}
}

template<class T>
FeatureMap<T>::~FeatureMap()
{
	_ref_count_dec();
}

template<class T>
FeatureMap<T>& FeatureMap<T>::operator=( const FeatureMap<T>& sl )
{
	_ref_count_dec();

	_width = sl._width;
	_height = sl._height;
	_padding_width = sl._padding_width;
	_padding_height = sl._padding_height;
	_p_start = sl._p_start;
	_ref_count = sl._ref_count;

	_ref_count_inc();
	
	return *this;
}

template<class T>
inline int FeatureMap<T>::width() const
{
	return _width - _padding_width;
}

template<class T>
inline int FeatureMap<T>::height() const
{
	return _height - _padding_height;
}

template<class T>
inline int FeatureMap<T>::rowstride() const
{
	return _width;
}

template<class T>
inline int FeatureMap<T>::padding_width() const
{
	return _padding_width;
}

template<class T>
inline int FeatureMap<T>::padding_height() const
{
	return _padding_height;
}

template<class T>
inline typename FeatureMap<T>::iterator FeatureMap<T>::begin( int x_skip, int y_skip, int padding_width, int padding_height )
{
	assert( padding_width <= _padding_width );
	assert( padding_height <= _padding_height );

	return iterator( _p_start + ( ( _padding_width - padding_width ) / 2 )
			+ ( ( _padding_height - padding_height ) / 2 ) * _width,
		x_skip,
		y_skip,
		_width,
		_padding_width );
}

template<class T>
inline typename FeatureMap<T>::iterator FeatureMap<T>::end( int x_skip, int y_skip, int padding_width, int padding_height )
{
	assert( padding_width <= _padding_width );
	assert( padding_height <= _padding_height );

	return iterator( _p_start + ( ( _padding_width - padding_width ) / 2 )
			+ ( ( _padding_height - padding_height ) / 2 ) * _width
			+ ( (int) ceil( (double) ( _height - _padding_height ) / y_skip ) * y_skip ) * _width,
		x_skip,
		y_skip,
		_width, 
		_padding_width );
}

template<class T>
std::ostream& FeatureMap<T>::put( std::ostream& s )
{
	s << this->width() << " "
		<< this->height() << endl;

	copy( this->begin(),
		this->end(),
		ostream_iterator<T>( s, "\n" ) );
	return s;
}

template<class T>
std::istream& FeatureMap<T>::get( std::istream& s )
{
	int width;
	int height;

	s >> width;
	if( width != this->width() )
	{
		throw( Exception( "width mismatch" ) );
	}
	s >> height;
	if( height != this->height() )
	{
		throw( Exception( "height mismatch" ) );
	}

	for( typename FeatureMap<T>::iterator i = this->begin();
		i != this->end();
		i++ )
	{
		if( !s.good() )
		{
			throw( Exception( "Error." ) );	
		}
		s >> *i;
	}
	return s;
}

template<class T>
void FeatureMap<T>::get( T* data )
{
	copy( data, data + width() * height(), begin() );
}

template<class T>
void FeatureMap<T>::put( T* data )
{
	copy( begin(), end(), data );
}

template<class T>
void FeatureMap<T>::fill_padding_with_noise( double level )
{
	for( int i = 0;
		i < _padding_height / 2;
		i ++ )
	{
		for( int j = 0;
			j < rowstride();
			j++ )
		{
			*( _p_start + i * rowstride() + j ) = rand() / (double) RAND_MAX * level;
			*( _p_start + ( _height - _padding_height + _padding_height / 2 + i ) * rowstride() + j ) = rand() / (double) RAND_MAX * level;
		}
	}

	for( int i = _padding_height / 2;
		i <= _padding_height / 2 + _height - _padding_height;
		i++ )
	{
		for( int j = 0;
			j < _padding_width / 2;
			j ++ )
		{
			*( _p_start + i * rowstride() + j ) = rand() / (double) RAND_MAX * level;
			*( _p_start + i * rowstride() + j + _padding_width / 2 + _width - _padding_width ) = rand() / (double) RAND_MAX * level;
		}
	}
}

#ifdef DEBUG
template<class T>
void FeatureMap<T>::debug_print()
{
	int x = 0;

	cout << "<FeatureMap>" << endl;
	for( pointer i = _p_start;
		i < _p_start + _width * _height;
		 i++, x++ )
	{
		if( x == ( _width ) )
		{
			x = 0;
			cout << endl;
		}
		cout << *i << " ";
	}
	cout << endl << "</FeatureMap>" << endl;
}
#endif

#endif //LAYERMAPPINGLIB_FEATUREMAPCODE_H
