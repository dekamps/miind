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

#ifndef LAYERMAPPINGLIB_NETWORKENSEMBLECODE_H
#define LAYERMAPPINGLIB_NETWORKENSEMBLECODE_H

#include "NetworkEnsemble.h"

using namespace LayerMappingLib;

template<class T>
typename NetworkEnsemble<T>::network& NetworkEnsemble<T>::add_network( const network& n )
{
	_v_network.push_back( n );

	return _v_network.back();
}

template<class T>
typename NetworkEnsemble<T>::iterator NetworkEnsemble<T>::begin()
{
	return _v_network.begin();
}

template<class T>
typename NetworkEnsemble<T>::iterator NetworkEnsemble<T>::end()
{
	return _v_network.end();
}

template<class T>
typename NetworkEnsemble<T>::network::iterator NetworkEnsemble<T>::begin( int i )
{
	return _v_network.at( i ).begin();
}

template<class T>
typename NetworkEnsemble<T>::network::iterator NetworkEnsemble<T>::end( int i )
{
	return _v_network.at( i ).end();
}

#ifdef DEBUG
template<class T>
void NetworkEnsemble<T>::debug_print()
{
	for( iterator i = _v_network.begin();
		i != _v_network.end();
		i++ )
	{
		i->debug_print();
	}
	
}
#endif //DEBUG

#endif //LAYERMAPPINGLIB_NETWORKENSEMBLECODE_H
