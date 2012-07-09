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

#ifndef LAYERMAPPINGLIB_FEATUREMAPITERATOR_H
#define LAYERMAPPINGLIB_FEATUREMAPITERATOR_H

#include <iterator>

namespace LayerMappingLib
{
	template<class T>
	class FeatureMap;

	/*! \class FeatureMapIterator
		\brief Iterator for itering through a FeatureMap

		FeatureMap resembles a matrix. The iterator iterates through this matrix row by row. Since the FeatureMap structure supports padding and skipping the iterator can iterate through the data in various ways. See the constructor documentation for more information.*/
	template<class T>
	class FeatureMapIterator : public std::iterator<std::forward_iterator_tag, T>
	{
		friend class FeatureMap<T>;

		public:
		typedef T value_type;
		typedef T* pointer;

		FeatureMapIterator<T>( const FeatureMapIterator<T>& i );

		value_type& operator*();
		
		FeatureMapIterator<T>& operator++();
		FeatureMapIterator<T> operator++( int );	

		bool operator!=( const FeatureMapIterator<T>& );
		bool operator==( const FeatureMapIterator<T>& );	

		private:
		/*! \brief Construct a FeatureMapIterator that supports skipping and padding.

			The iterator points to the data position defined by position. 
			\param position The data position.
			\param x_skip The data position the iterator skips in x direction.
			\param y_skip The data position the iterator skips in y direction.
			\param row_stride The row_stride of the FeatureMap to iterate through.
			\param padding_width The padding width taken into account.*/
		FeatureMapIterator( T* position,
			int x_skip,
			int y_skip,
			int row_stride,
			int padding_width = 0);

		T* _p_position;
		T* _p_row_start;

		int _x_skip;
		int _y_skip;
		int _row_stride;
		int _padding_width;
	};
}

#endif //LAYERMAPPINGLIB_FEATUREMAPITERATOR_H
