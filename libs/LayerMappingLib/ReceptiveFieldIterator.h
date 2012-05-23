// Copyright (c) 2005 - 2010 Marc de Kamps, Johannes Drever, Melanie Dietz
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

#ifndef LAYERMAPPINGLIB_RECEPTIVEFIELDITERTOR_H
#define LAYERMAPPINGLIB_RECEPTIVEFIELDITERTOR_H

#include <iterator>

#include <assert.h>

namespace LayerMappingLib
{
	template<class T>
	class ReceptiveField;

	template<class T>
	class ReceptiveFieldIterator : public std::iterator<std::forward_iterator_tag, T>
	{
		friend class ReceptiveField<T>;

		public:
		typedef T value_type;
		typedef typename std::iterator_traits<std::iterator<std::forward_iterator_tag, T> >::difference_type difference_type;

		ReceptiveFieldIterator();

		value_type& operator*();

		ReceptiveFieldIterator<T>& operator++();
		ReceptiveFieldIterator<T> operator++( int );
		ReceptiveFieldIterator<T> operator+( int );
		ReceptiveFieldIterator<T>& operator+=( int );

		bool operator!=( const ReceptiveFieldIterator<T>& i );
		bool operator==( const ReceptiveFieldIterator<T>& i );

		difference_type operator-( const ReceptiveFieldIterator<T>& i );

		private:
		ReceptiveFieldIterator( T* p, int rf_widht, int y_inc );

		T* _p_position;
		T* _p_row_start;

		int _rf_width;
		int _y_inc;
	};
}

#endif //LAYERMAPPINGLIB_RECEPTIVEFIELDITERTOR_H
