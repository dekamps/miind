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

#ifndef LAYERMAPPINGLIB_FEATUREMAP_H
#define LAYERMAPPINGLIB_FEATUREMAP_H

#include "FeatureMapIteratorCode.h"
#include "Exception.h"

#include <assert.h>
#include <iostream>
#include <iterator>

#ifdef DEBUG
#include <iostream>
#endif //DEBUG

#include <math.h>

namespace LayerMappingLib
{
	/*! \class FeatureMap
		\brief Data sructure for a FeatureMap. Resembles a padded two dimensional array.

		A FeatureMap resembles a two dimensional padded array. The size of the padding is defined by construction and the padding is filled with zeros. 

		FeatureMap acts as a reference linked shared pointer. This means that the data is allocated once, when the default constructor is called. If the copy constructor or assignment operator is used the data is not copied. To keep track of the references and for garbage collection the number of references is counted.

		The elements in the feature map can be accessed via iterators. The iterators can be parameterized to iterate through the feature map with different skip sizes. The amount of padding taken into account can be parameterized as well. TODO ref to test code.*/
	template<class T>
	class FeatureMap
	{
		public:
		typedef FeatureMapIterator<T> iterator;
		typedef T value_type;
		typedef T* pointer;

		/*! \brief Default Constructor 

			Constructs a"NULL" feature map */
		FeatureMap();
		/*! \brief Constructor

			Construct a feature map.
			\param width The width of the feature map
			\param height The height of the feature map
			\param padding_width The width of the padding of the feature map.
			\param padding_height The height of the padding of the feature map*/
		FeatureMap( int width, int height, int padding_width, int padding_height );
		/*! \brief Copy Constructor
			
			Since FeatureMap implements a reference linked shared pointer the data of the feature map is not copied. */
		FeatureMap( const FeatureMap<T>& sl );
		~FeatureMap();

		/*! \brief The assignment operator

			Since FeatureMap implements a reference linked shared pointer the data of the feature map is not copied.*/
		FeatureMap<T>& operator=( const FeatureMap<T>& sl );

		/*! \brief Iterator that points to begin.
			\param x_skip Specifies how many values the iterator skips in x.
			\param y_skip Specifies how many values the iterator skips in y.
			\param padding_width Skip (padding_width / 2) rows.
			\param padding_height Skip (padding_height / 2) columns.
			\return An iterator that points to the "first" element, where "first" depends on padding_width and padding_height.*/
		iterator begin( int x_skip = 1, int y_skip = 1, int padding_width = 0, int padding_height = 0 );
		/*! \brief Iterator that points to end.
			\param x_skip Specifies how many values the iterator skips in x.
			\param y_skip Specifies how many values the iterator skips in y.
			\param padding_width Skip (padding_width / 2) rows.
			\param padding_height Skip (padding_height / 2) columns.
			\return An iterator that points past the "last" element, where "last" depends on padding_width and padding_height.*/
		iterator end( int x_skip = 1, int y_skip = 1, int padding_width = 0, int padding_height = 0 );

		/*! \brief The width of the FeatureMap, ignoring padding. */
		int width() const;
		/*! \brief The height of the FeatureMap, ignoring padding. */
		int height() const;
		/*! \brief The width of the FeatureMap including padding. */
		int rowstride() const;

		/*! \brief The width of the padding. */
		int padding_width() const;
		/*! \brief The height of the padding. */
		int padding_height() const;

		/*! \brief Put the FeatureMap to a stream.*/
		std::ostream& put( std::ostream& s );
		/*! \brief Get the FeatureMap from a stream.*/
		std::istream& get( std::istream& s );

		/*! \brief Put the FeatureMap to a raw array.

			No range checking is made at all. */
		void get( T* data );
		/*! \brief Get the FeatureMap from a raw array.

			No range checking is made at all. */
		void put( T* data );

		//TODO A function fill_padding with a functional that specifies how the padding is filled would be nicer!
		void fill_padding_with_noise( double level );

		#ifdef DEBUG
		void debug_print();
		#endif //DEBUG

		private:
		void _ref_count_inc();
		void _ref_count_dec();
		void _deallocate();

		T* _p_start;

		int* _ref_count;

		int _width;
		int _height;

		int _padding_width;
		int _padding_height;
	};

	template<class T>
	std::ostream& operator<<( std::ostream& s, FeatureMap<T>& l )
	{
		l.put( s );
		return s;
	}
	template<class T>
	std::istream& operator>>( std::istream& s, FeatureMap<T>& l )
	{
		l.get( s );
		return s;
	}
}

#endif //LAYERMAPPINGLIB_FEATUREMAP_H
