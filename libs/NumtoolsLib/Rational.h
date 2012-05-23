// Copyright (c) 2005 - 2009 Marc de Kamps, Korbinian Trumpp
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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_NUMTOOLSLIB_RATIONAL_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_RATIONAL_INCLUDE_GUARD

// Date:   29-06-2004
// Author: Korbinian Trumpp

#include "NumtoolsLibException.h"
#include <iostream>

using std::istream;
using std::ostream;

namespace NumtoolsLib
{
	/// A basic class for a rational representation without doubles and their rounding errors
	class Rational
	{
		public:

			/// constructor for a rational with numerator and denominator as arguments
			Rational( int = 0, int = 1 );

			/// Destructor.
			virtual ~Rational();

			/// Returns the int value of the rational without remainder
			int getValue() const{ return _numerator / _denominator; }
			/// Returns the remainder of the rational
			int getRemainder() const{ return _numerator % _denominator; }
			/// Returns the "true" value of the rational as double but with possible rounding and casting errors
			double getTrueValueThroughCast();

			/// Set a new value for the whole rational
			void set( int, int );

			int _numerator;
			unsigned int _denominator;

			Rational& operator=( const Rational& );
			Rational& operator+=( const Rational& );
			Rational& operator++();
			Rational& operator-=( const Rational& );
			Rational& operator--();
	};

	ostream& operator<<( ostream&, const Rational& );
	istream& operator>>( istream&, Rational& );

	bool operator==( const Rational&, const Rational& );
	bool operator!=( const Rational&, const Rational& );
	bool operator<( const Rational&, const Rational& );
	bool operator<=( const Rational&, const Rational& );
	bool operator>( const Rational&, const Rational& );
	bool operator>=( const Rational&, const Rational& );

	Rational operator+( const Rational&, const Rational& );
	Rational operator+( const Rational&, int );
	Rational operator+( int, const Rational& );
	Rational operator-( const Rational&, const Rational& );
	Rational operator-( const Rational&, int );
	Rational operator-( int, const Rational& );
	Rational operator*( const Rational&, const Rational& );
	Rational operator*( const Rational&, int );
	Rational operator/( const Rational&, const Rational& );
	Rational operator/( int, const Rational& );
	int operator%( int, const Rational& );

	int ggt( int, int );

} // end of Numtools


#endif // include guard
