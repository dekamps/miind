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

#ifndef LAYERMAPPINGLIB_FUNCTIONFACTORYCODE_H
#define LAYERMAPPINGLIB_FUNCTIONFACTORYCODE_H

#ifdef WIN32
#pragma warning(disable: 4996)
#endif 

#include "FunctionFactory.h"

using namespace LayerMappingLib;

template<class Function>
Function* FunctionFactory<Function>::get_function( iostream& s )
{
	string function;
	s >> function;

	//Register your function here
	if( !function.compare( MEAN ) )
	{
		return new Mean<vector_list>;
	}

	else if( !function.compare( STANDARD_DEVIATION ) )
	{
		return new StandardDeviation<vector_list>;
	}

	else if( !function.compare( MIN ) )
	{
		return new Min<vector_list>;
	}

	else if( !function.compare( MAX ) )
	{
		return new Max<vector_list>;
	}

	else if( !function.compare( SUM ) )
	{
		return new Sum<vector_list>;
	}

	else if( !function.compare( PRODUCT ) )
	{
		return new Product<vector_list>;
	}

	else if( !function.compare( COMPOSITE_FEATURE ) )
	{
		return new CompositeFeature<vector_list>;
	}

	else if( !function.compare( ARGMAX ) )
	{
		return new ArgMax<vector_list>;
	}

	else if( !function.compare( IDENTITY ) )
	{
		return new Identity<vector_list>;
	}
	else if( !function.compare( COMBINE ) )
	{
		return new Combine<vector_list>;
	}
	else if( !function.compare( SCALE ) )
	{
		return new Scale<vector_list>;
	}
	else if( !function.compare( PRODUCT_COMPLEX_REAL ) )
	{
		return new ProductComplexReal<vector_list>;
	}
	else if( !function.compare( PRODUCT_COMPLEX_IMAG ) )
	{
		return new ProductComplexImag<vector_list>;
	}
	#ifdef HAVE_FFTW
	else if( !function.compare( FFTR_ ) || !function.compare( FFTC_ ) )
	{
		string t;

		fft_type type;
		int sign;

		s >> t;
		if( !t.compare( FFT_TYPE_REAL ) )
		{
			type = REAL;
		}
		else if( !t.compare( FFT_TYPE_IMAG ) )
		{
			type = IMAG;
		}
		else
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss.str( "Unkown option for FFT: " );
			ss << t;
			throw Exception( ss.str() );
		}

		s >> t;
		if( !t.compare( FFT_FORWARD ) )
		{
			sign = FFTW_FORWARD;
		}
		else if( !t.compare( FFT_BACKWARD ) )
		{
			sign = FFTW_BACKWARD;
		}
		else
		{
			stringstream ss( stringstream::in | stringstream::out );
			ss.str( "Unkown option for FFT: " );
			ss << t;
			throw Exception( ss.str() );
		}

		int width;
		int height;

		s >> width;
		s >> height;
		if( !function.compare( FFTR_ ) )
		{
			return new FFTR<vector_list>( type, sign, width, height );
		}
		else
		{
			return new FFTC<vector_list>( type, sign, width, height );
		}
	}
	#endif //HAVE_FFTW

	else if( !function.compare( PERCEPTRON ) )
	{
		int nr_weights;

		s >> nr_weights;

		vector<double> weights( nr_weights );

		for( vector<double>::iterator i = weights.begin();
			i != weights.end();
			i++ )
		{
			s >> *i;
		}
//		return new Perceptron<vector_list>( weights );
	}
	else if( !function.compare( SIMPLE_CELL_RESPONSE ) )
	{
		int width;
		int height;
	
		double sigma;

		s >> sigma;

		s >> width;
		s >> height;
		vector<vector<double> > m;
	
		for( int x = 0; x < width; x++ )
		{
			vector<double> r;
			for( int y = 0; y < height; y++ )
			{
				double d;
				s >> d;
				r.push_back( d );
			}
			m.push_back( r );	
		}
		return new SimpleCellResponse<vector_list, vector<vector<double> > >( width, height, m, sigma );
	}
	else if( !function.compare( CONVOLUTION ) )
	{
		int width;
		int height;
	
		s >> width;
		s >> height;
		vector<vector<double> > m;
	
		for( int x = 0; x < width; x++ )
		{
			vector<double> r;
			for( int y = 0; y < height; y++ )
			{
				double d;
				s >> d;
				r.push_back( d );
			}
			m.push_back( r );	
		}
		return new Convolution<vector_list, vector<vector<double> > >( width, height, m );
	}
	else
	{
		stringstream ss( stringstream::in | stringstream::out );
		ss.str( "Unkown function: " );
		ss << function;
		throw Exception( ss.str() );
	}
	return FunctionFactory<Function>::empty_function();
}

template<class Function>
Function* FunctionFactory<Function>::empty_function()
{
	return static_cast<Function*>( NULL );
}

#endif //LAYERMAPPINGLIB_FUNCTIONFACTORYCODE_H
