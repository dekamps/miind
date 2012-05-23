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

#ifndef LAYERMAPPINGLIB_FUNCTIONS_ARGMAXCODE_H
#define LAYERMAPPINGLIB_FUNCTIONS_ARGMAXCODE_H

#include "ArgMax.h"

using namespace LayerMappingLib;

template<class VectorList>
inline void ArgMax<VectorList>::operator()( vector_iterator input_begin, vector_iterator input_end,
			iterator output_begin, iterator output_end )
{
	//TODO
	int argmax = 0;
	value_type maximum = -std::numeric_limits<value_type>::infinity();

	for( vector_iterator i = input_begin;
		i != input_end;
		i++ )
	{
		int t = 0;
		value_type temp = -std::numeric_limits<value_type>::infinity();
		for( iterator j = i->begin();
			j != i->end();
			j++, t++ )
		{
			if( temp < *j )
			{
				temp = *j;
				argmax = t;
			}
		}
		maximum = max( temp, maximum );
	}
	fill( output_begin,
		output_end,
		0.0 );
	*(output_begin + argmax) = maximum;
}

#ifdef DEBUG
template<class VectorList>
void ArgMax<VectorList>::debug_print() const
{
	cout << "<ArgMax>" << endl << "</ArgMax>" << endl;
}
#endif //DEBUG

#endif //LAYERMAPPINGLIB_FUNCTIONS_ARGMAXCODE_H
