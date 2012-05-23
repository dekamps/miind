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

#ifndef LAYERMAPPINGLIB_LAYERMAPPINGUTIL
#define LAYERMAPPINGLIB_LAYERMAPPINGUTIL

/*! \file This file contains some functions that are used in LayerMappingLib but do not belong to the library. These should be
	taken to UtilLib or replaced by functions from e.g. boost.
*/

#include <vector>

using namespace std;

namespace LayerMappingLib
{
	/* JO code starts here*/
	template<class T>
	struct incrementer
	{
		incrementer( T v, T i ) : _v( v ), _i ( i ){};
		
		T operator()()
		{
			T r = _v;
			_v += _i;
			return r;
		}
	
		T _v;
		T _i;
	};
	
	/*! \brief Returns a list of all possible combinations.

		A list of all possible combinations of values between [0, v[ with length l. The entries grow exponentional in l.*/
	vector<vector<int> > combinations( int v, int l );

	/*! \brief Returns a list of all possible combinations.

		A list of all possible combinations of values between [0, v[ with length l. The entries grow exponentional in l.*/
	//vector<vector<int> > combinations_iterative( int v, int l );

#define PI 3.14159265
	/*! \brief Second derivat gaussian filter as used in the original HMAX model.
		TODO -> ImageProcessing? */
	template <class T>
	vector<vector<T> > second_derivat_gaussian( T sigDivisor, int width, int height, double orientation )
	{
		// calculate parameters for gauss
		int fSizH = ( width + 1 - width % 2 ) / 2;
		T sigmaq = pow ( width / sigDivisor, 2.0 );
		T orientation_RAD = orientation* ( PI ) / 180.0;
		T sinus_rot = sin ( orientation_RAD );
		T cos_rot = cos ( orientation_RAD );
	
	
		vector<vector<T> > filter;
		for( int i = 0; i < height; i++ )
		{
			filter.push_back( vector<T>( width ) );
		}
	
		T sum_column = 0.0;
	
		//calculate filter values with center at (fSizH,fSizH)
		for ( int y = -fSizH; y <= fSizH; ++y )
		{	// Zeilen
			int y_old = y + fSizH;
	
			T sum_row = 0.0;
			for ( int x = -fSizH; x <= fSizH; ++x )
			{		// Spalten
				int x_old = x+fSizH;
	
				// calculate the pixel's rotated position
				T x_new = cos_rot * x - sinus_rot * y;
				T y_new = sinus_rot * x + cos_rot * y;
				// calculate gaussian's second derivate for the rotated position
				// R(f(x,y)) = f(R(x),R(y))
				//filter(spalte, zeile) !!!
	
				filter.at( x_old ).at( y_old ) = ( pow ( y_new,2.0 ) /sigmaq-1.0 )
					/ sigmaq *
					exp ( - ( pow ( x_new,2.0 ) + pow ( y_new,2.0 ) ) / ( 2.0*sigmaq ) );
				//(pow(row,2)/sigmaq-1)/sigmaq*exp(-(pow(column,2)+pow(row,2))/(2*sigmaq))
	
				sum_row += filter.at( x_old ).at( y_old );
			}
			sum_column += sum_row;
		}
	
		T mean = sum_column / ( height * width ) ;
		T square_sum = 0.0;
	
		for ( int i = 0; i < width; ++i )
		{
			for ( int j = 0; j < height; ++j )
			{
				filter.at( i ).at( j ) = filter.at( i ).at( j ) - mean;
				square_sum += pow ( filter.at( i ).at( j ), 2.0 );
			}
		}
	
		T sqrt_sum = sqrt ( square_sum );
		//_out << "sqrt_sum: " << sqrt_sum << std::endl;
		for ( int i = 0;  i < width; ++i )
		{
			for ( int j = 0; j < height; ++j )
			{
				filter.at( i ).at( j ) = ( filter.at( i ).at( j ) / sqrt_sum );
			}
		}
		//everything needed is done
		return filter;
	}

	template<class T>
	vector<vector<T> > gaussian( int width, int height, T variance )
	{
		vector<vector<T> > r;

		T scale_factor = 1 / ( 2 * PI * ( 2 * variance ) );
		for( int y = 0;
			y != height;
			y++ )
		{
			r.push_back( vector<T>( width ) );
			for( int x = 0; x < width; x++ )
			{
				T v = scale_factor * exp( - 0.5 * ( ( x * x * ( 1 / v ) ) + ( y * y * ( 1 / v  )) ) );
				r.at( y ).at( x ) = v;
			}
		}
		return r;
	}

	/*! \brief Gabor filter, as used by the Serre extension of the HMAX model.

		TODO -> ImageProcessing?*/
	template<class T>
	vector<vector<T> > gabor( int width, int height, T sigma, T lambda, T theta, T gamma )
	{
		vector<vector<T> > r( height ); //TODO test if correct
		fill( r.begin(), r.end(), vector<T>( width ) );
		for( int x = 0;
			x < width;
			x++ )
		{
			for( int y = 0;
				y < height;
				y++ )
			{
				T x_0 = x * cos( theta ) + y * sin( theta );
				T y_0 = -x * sin( theta ) + y * cos( theta );

				r.at( y ).at( x ) =
					exp( -( ( ( x_0 * x_0 ) + ( gamma * gamma * y_0 * y_0 ) )
						/ ( 2 * sigma * sigma ) ) )
					* cos( ( 2 * PI ) / ( lambda ) * x_0 );
			}
		}

		return r;
	}
	/* JO code ends here */

} // end of Util 

#endif //LAYERMAPPINGLIB_LAYERMAPPINGUTIL
