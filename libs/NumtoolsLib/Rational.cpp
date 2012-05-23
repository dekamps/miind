// Copyright (c) 2005 - 2008 Marc de Kamps
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
#include "Rational.h"

using namespace NumtoolsLib;

Rational::Rational( int num, int den ) :
	_numerator( num ),
	_denominator( den )
{
	if( den == 0 )
		throw NumtoolsException("Rational denominator may not be zero!!!"); 
}

//---------------------------------------------------------------------------

Rational::~Rational(){}

//---------------------------------------------------------------------------

double Rational::getTrueValueThroughCast()
{
	return (double)_numerator / (double)_denominator;
}

//---------------------------------------------------------------------------

void Rational::set( int num, int den )
{
	if( den == 0 )
		throw NumtoolsException("Rational denominator may not be zero!!!"); 
	_numerator = num;
	_denominator = den;
}

//---------------------------------------------------------------------------

Rational& Rational::operator=( const Rational& rat )
{
	_numerator = rat._numerator;
	_denominator = rat._denominator;
	return *this;
}

//---------------------------------------------------------------------------

Rational& Rational::operator+=( const Rational& rat )
{
	if( _denominator == rat._denominator )
		_numerator += rat._numerator;
	else
	{
		_denominator *= rat._denominator;
		_numerator = _numerator * rat._denominator + rat._numerator * _denominator;
		int div = ggt( abs(_numerator), _denominator );
		_denominator /= div;
		_numerator /= div;
	}
	return *this;
} 

//---------------------------------------------------------------------------

Rational& Rational::operator++()
{
	_numerator += _denominator;
	return *this;
} 

//---------------------------------------------------------------------------

Rational& Rational::operator-=( const Rational& rat )
{
	if( _denominator == rat._denominator )
		_numerator -= rat._numerator;
	else
	{
		_denominator *= rat._denominator;
		_numerator = _numerator * rat._denominator - rat._numerator * _denominator;
		int div = ggt( abs(_numerator), _denominator );
		_denominator /= div;
		_numerator /= div;
	}
	return *this;
} 

//---------------------------------------------------------------------------

Rational& Rational::operator--()
{
	_numerator -= _denominator;
	return *this;
} 

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

ostream& NumtoolsLib::operator<<( ostream& s, const Rational& rat )
{
	s << rat._numerator << " / ";
	s << rat._denominator;
	return s;
}

//---------------------------------------------------------------------------

istream& NumtoolsLib::operator>>( istream& s, Rational& rat )
{
	s >> rat._numerator;
	char line;
	s >> line;
	s >> rat._denominator;
	return s;
}

//---------------------------------------------------------------------------

bool NumtoolsLib::operator==( const Rational& rat1, const Rational& rat2 )
{
	return ( rat1.getValue() == rat2.getValue() && rat1.getRemainder()*rat2._denominator == rat2.getRemainder()*rat1._denominator );
}

//---------------------------------------------------------------------------

bool NumtoolsLib::operator!=( const Rational& rat1, const Rational& rat2 )
{
	return ( rat1.getValue() != rat2.getValue() || rat1.getRemainder()*rat2._denominator != rat2.getRemainder()*rat1._denominator );
}

//---------------------------------------------------------------------------

bool NumtoolsLib::operator<( const Rational& rat1, const Rational& rat2 )
{
	if( rat1.getValue() < rat2.getValue() )
		return true;
	else
		return ( rat1.getValue() == rat2.getValue() && rat1.getRemainder()*rat2._denominator < rat2.getRemainder()*rat1._denominator );
}

//---------------------------------------------------------------------------

bool NumtoolsLib::operator<=( const Rational& rat1, const Rational& rat2 )
{
	if( rat1.getValue() < rat2.getValue() )
		return true;
	else
		return ( rat1.getValue() == rat2.getValue() && rat1.getRemainder()*rat2._denominator <= rat2.getRemainder()*rat1._denominator );
}

//---------------------------------------------------------------------------

bool NumtoolsLib::operator>( const Rational& rat1, const Rational& rat2 )
{
	if( rat1.getValue() > rat2.getValue() )
		return true;
	else
		return ( rat1.getValue() == rat2.getValue() && rat1.getRemainder()*rat2._denominator > rat2.getRemainder()*rat1._denominator );
}

//---------------------------------------------------------------------------

bool NumtoolsLib::operator>=( const Rational& rat1, const Rational& rat2 )
{
	if( rat1.getValue() > rat2.getValue() )
		return true;
	else
		return ( rat1.getValue() == rat2.getValue() && rat1.getRemainder()*rat2._denominator >= rat2.getRemainder()*rat1._denominator );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator+( const Rational& rat1, const Rational& rat2 )
{
	int num = 0, den = 0;
	if( rat1._denominator == rat2._denominator )
	{
		den = rat1._denominator;
		num = rat1._numerator + rat2._numerator;
	}
	else
	{
		den = rat1._denominator * rat2._denominator;
		num = rat1._numerator * rat2._denominator + rat2._numerator * rat1._denominator;
		int div = ggt( abs(num), den );
		den /= div;
		num /= div;
	}
	return Rational( num, den );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator+( const Rational& rat, int i )
{
	int num = 0, den = 0;
	num = rat._numerator + i * rat._denominator;
	den = rat._denominator;
	return Rational( num, den );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator+( int i, const Rational& rat )
{
	int num = 0, den = 0;
	num = rat._numerator + i * rat._denominator;
	den = rat._denominator;
	return Rational( num, den );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator-( const Rational& rat1, const Rational& rat2 )
{
	int num = 0, den = 0;
	if( rat1._denominator == rat2._denominator )
	{
		den = rat1._denominator;
		num = rat1._numerator - rat2._numerator;
	}
	else
	{
		den = rat1._denominator * rat2._denominator;
		num = rat1._numerator * rat2._denominator - rat2._numerator * rat1._denominator;
		int div = ggt( abs(num), den );
		den /= div;
		num /= div;
	}
	return Rational( num, den );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator-( const Rational& rat, int i )
{
	int num = 0, den = 0;
	num = rat._numerator - i * rat._denominator;
	den = rat._denominator;
	return Rational( num, den );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator-( int i, const Rational& rat )
{
	int num = 0, den = 0;
	num = i * rat._denominator - rat._numerator;
	den = rat._denominator;
	return Rational( num, den );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator*( const Rational& rat1, const Rational& rat2 )
{
	int num = 0, den = 0;
	num = rat1._numerator * rat2._numerator;
	den = rat1._denominator * rat2._denominator;
	int div = ggt( abs(num), den );
	den /= div;
	num /= div;
	return Rational( num, den );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator*( const Rational& rat, int i )
{
	return Rational( rat._numerator * i, rat._denominator );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator/( const Rational& rat1, const Rational& rat2 )
{
	int num = 0, den = 0;
	num = rat1._numerator * rat2._denominator;
	den = rat1._denominator * rat2._numerator;
	int div = ggt( abs(num), den );
	den /= div;
	num /= div;
	return Rational( num, den );
}

//---------------------------------------------------------------------------

Rational NumtoolsLib::operator/( int i, const Rational& rat )
{
	int num = 0, den = 0;
	num = i * rat._denominator;
	den = rat._numerator;
	int div = ggt( abs(num), den );
	den /= div;
	num /= div;
	return Rational( num, den );
}

//---------------------------------------------------------------------------

int NumtoolsLib::operator%( int i, const Rational& rat )
{
	if( rat.getRemainder() == 0 )
		return i % rat.getValue();
	if( rat.getValue() == 0 && ggt( rat._numerator, rat._denominator ) == rat._numerator )
		return 0;

	throw NumtoolsException("Error in modulo, code should never get here");
}

//---------------------------------------------------------------------------

int NumtoolsLib::ggt( int a, int b )
{
	if (b==0)
		return a;
	else
		return ggt(b, a%b);
}
