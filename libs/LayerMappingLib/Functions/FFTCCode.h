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

#ifdef HAVE_FFTW

#ifndef LAYERMAPPINGLIB_FUNCTIONS_FFTCCODE
#define LAYERMAPPINGLIB_FUNCTIONS_FFTCCODE

#include "FFTC.h"

using namespace LayerMappingLib;

template<class VectorList>
FFTC<VectorList>::FFTC( fft_type t, int sign, int width, int height ) :
				_type( t ),
				_sign( sign ),
				_width( width ),
				_height( height )
{
}
#include <iostream>
using namespace std;

template<class VectorList>
inline void FFTC<VectorList>::operator()( vector_iterator input_begin, vector_iterator input_end,
			iterator output_begin, iterator output_end )
{
	assert( ( input_end - input_begin ) == 2 );
	assert( _type == REAL || _type == IMAG );

	int N = (int) ( input_begin->end() - input_begin->begin() );

	assert( N == _width * _height );

	fftw_complex* in = (fftw_complex*) fftw_malloc( sizeof(fftw_complex) * N );
	fftw_complex* out = (fftw_complex*) fftw_malloc( sizeof(fftw_complex) * N );

	fftw_complex* f = in;
	vector_iterator in2 = input_begin;
	in2++;
	iterator j = in2->begin();
	for( iterator i = input_begin->begin();
		i != input_begin->end();
		i++, j++, f++ )
	{
		(*f)[ 0 ] = *i;
		(*f)[ 1 ] = *j;
	}
	assert( f == in + N );
	assert( j == in2->end() );

	fftw_plan p = fftw_plan_dft_2d( _width, _height, in, out, _sign, FFTW_ESTIMATE );

	fftw_execute( p );

	double scale_factor = 1;// / sqrt( N );
	if( _type == REAL )
	{
		iterator o = output_begin;
		for( fftw_complex* i = out;
			i < out + N;
			i++, o++ )
		{
			*o = (*i)[ 0 ] * scale_factor;
		}
		assert( o == output_end );
	}
	else if( _type == IMAG )
	{
		iterator o = output_begin;
		for( fftw_complex* i = out;
			i < out + N;
			i ++, o++ )
		{
			*o = (*i)[ 1 ] * scale_factor;
		}
		assert( o == output_end );
	}

	fftw_destroy_plan( p );

	fftw_free( in );
	fftw_free( out );
}

#ifdef DEBUG
#include <iostream>
using namespace std;

template<class VectorList>
void FFTC<VectorList>::debug_print() const
{
	cout << "<FFTC>" << _type << "</FFTC>" << endl;
}
#endif //DEBUG

#endif //LAYERMAPPINGLIB_FUNCTIONS_FFTCCODE

#endif //HAVE_FFTW
