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

#ifndef LAYERMAPPINGLIB_FUNCTIONS_STANDARTDEVIATIONCODE_H
#define LAYERMAPPINGLIB_FUNCTIONS_STANDARTDEVIATIONCODE_H

#include "StandardDeviation.h"

template<class VectorList>
inline void StandardDeviation<VectorList>::operator()( vector_iterator input_begin, vector_iterator input_end,
			iterator output_begin, iterator output_end )
{
	value_type mean;

	int c = 0;
	value_type acc = 0;

	for( vector_iterator v = input_begin;
		v != input_end;
		v++ )
	{
		for( iterator i = v->begin();
			i != v->end();
			i++, c++ )
		{
			acc += ( *i );
		}
	}
	mean = acc / c;

	acc = 0;

	for( vector_iterator v = input_begin;
		v != input_end;
		v++ )
	{
		for( iterator i = v->begin();
			i != v->end();
			i++ )
		{
			acc += ( *i - mean ) * ( *i - mean );
		}
	}

	value_type result = pow( acc / c, 0.5 );
	fill( output_begin,
		output_end,
		result );
}

#ifdef DEBUG
#include <iostream>
using namespace std;

template<class VectorList>
void StandardDeviation<VectorList>::debug_print() const
{
	cout << "<StandardDeviation>" << "</StandardDeviation>" << endl;
}
#endif //DEBUG

#endif //LAYERMAPPINGLIB_FUNCTIONS_STANDARTDEVIATIONCODE_H
