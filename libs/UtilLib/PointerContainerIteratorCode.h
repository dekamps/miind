// Copyright (c) 2005 - 2010 Johannes Drever
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
#ifndef UTILLIB_POINTERCONTAINERITERATORCODE_H
#define UTILLIB_POINTERCONTAINERITERATORCODE_H

#include "PointerContainerIterator.h"

using namespace UtilLib;

template<class PointerIterator>
PointerContainerIterator<PointerIterator>::PointerContainerIterator( PointerIterator i_p ) : _i_p( i_p )
{
}

template<class PointerIterator>
typename  PointerContainerIterator<PointerIterator>::value_type& PointerContainerIterator<PointerIterator>::operator*()
{
	return **_i_p;
}

template<class PointerIterator>
typename PointerContainerIterator<PointerIterator>::pointer PointerContainerIterator<PointerIterator>::operator->()
{
	return &(operator*());
}

template<class PointerIterator>
typename  PointerContainerIterator<PointerIterator>::pointer PointerContainerIterator<PointerIterator>::operator&()
{
	return *_i_p;
}

template<class PointerIterator>
PointerContainerIterator<PointerIterator>& PointerContainerIterator<PointerIterator>::operator++()
{
	_i_p++;
	
	return *this;
}

template<class PointerIterator>
PointerContainerIterator<PointerIterator> PointerContainerIterator<PointerIterator>::operator++( int )
{
	PointerContainerIterator<PointerIterator> r = *this;
	++*this;
	return r;
}

template<class PointerIterator>
bool PointerContainerIterator<PointerIterator>::operator!=( const PointerContainerIterator<PointerIterator>& i )
{
	return _i_p != i._i_p;
}

template<class PointerIterator>
bool PointerContainerIterator<PointerIterator>::operator<( const PointerContainerIterator<PointerIterator>& i )
{
	return _i_p < i._i_p;
}

template<class PointerIterator>
bool PointerContainerIterator<PointerIterator>::operator>( const PointerContainerIterator<PointerIterator>& i )
{
	return _i_p > i._i_p;
}

template<class PointerIterator>
bool PointerContainerIterator<PointerIterator>::operator<=( const PointerContainerIterator<PointerIterator>& i )
{
	return _i_p <= i._i_p;
}

template<class PointerIterator>
bool PointerContainerIterator<PointerIterator>::operator>=( const PointerContainerIterator<PointerIterator>& i )
{
	return _i_p >= i._i_p;
}

template<class PointerIterator>
bool PointerContainerIterator<PointerIterator>::operator==( const PointerContainerIterator<PointerIterator>& i )
{
	return _i_p == i._i_p;
}

template<class PointerIterator>
PointerContainerIterator<PointerIterator> PointerContainerIterator<PointerIterator>::operator+( int n )
{
	PointerContainerIterator<PointerIterator> r = *this;

	r._i_p = r._i_p + n;

	return r;
}

template<class PointerIterator>
PointerContainerIterator<PointerIterator>& PointerContainerIterator<PointerIterator>::operator+=( int n )
{
	_i_p = _i_p + n;

	return *this;
}

template<class PointerIterator>
PointerContainerIterator<PointerIterator> PointerContainerIterator<PointerIterator>::operator-( int n )
{
	PointerContainerIterator<PointerIterator> r = *this;

	r._i_p = r._i_p - n;

	return r;
}

template<class PointerIterator>
PointerContainerIterator<PointerIterator>& PointerContainerIterator<PointerIterator>::operator-=( int n )
{
	_i_p = _i_p - n;

	return *this;
}

template<class PointerIterator>
int PointerContainerIterator<PointerIterator>::operator+( const PointerContainerIterator<PointerIterator>&  o )
{
	return _i_p + o;
}

template<class PointerIterator>
int PointerContainerIterator<PointerIterator>::operator-( const PointerContainerIterator<PointerIterator>& o )
{
	return _i_p - o;
}

#endif //UTILLIB_POINTERCONTAINERITERATORCODE_H
