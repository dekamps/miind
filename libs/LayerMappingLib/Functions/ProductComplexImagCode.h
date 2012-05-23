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

#ifndef LAYERMAPPINGLIB_FUNCTIONS_PRODUCTCOMPLEXIMAGCODE_H
#define LAYERMAPPINGLIB_FUNCTIONS_PRODUCTCOMPLEXIMAGCODE_H

#include "ProductComplexImag.h"

template<class VectorList>
inline void ProductComplexImag<VectorList>::operator()( vector_iterator input_begin, vector_iterator input_end,
			iterator output_begin, iterator output_end )
{
	vector_iterator ib = input_begin;
	iterator a = ib->begin();
	iterator a_end = ib->end(); //TODO this code is only for debug asserts
	ib++;
	iterator b = ib->begin();
	iterator b_end = ib->end();
	ib++;
	iterator c = ib->begin();
	iterator c_end = ib->end();
	ib++;
	iterator d = ib->begin();
	iterator d_end = ib->end();
	ib++;
	assert( ib == input_end );

	for( iterator o = output_begin;
		o != output_end;
		o++,
			a++, b++, c++, d++ )
	{
		*o = ( (*a) * (*d) ) + ( (*b) * (*c) );
	}
	assert( a == a_end );
	assert( b == b_end );
	assert( c == c_end );
	assert( d == d_end );
}

#ifdef DEBUG
#include <iostream>
using namespace std;

template<class VectorList>
void ProductComplexImag<VectorList>::debug_print() const
{
	cout << "<ProductComplexImag>" << endl << "</ProductComplexImag>" << endl;
}
#endif //DEBUG
#endif //LAYERMAPPINGLIB_FUNCTIONS_PRODUCTCOMPLEXIMAGCODE_H
