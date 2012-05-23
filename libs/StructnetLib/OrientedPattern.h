// Copyright (c) 2005 - 2009 Marc de Kamps
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
#ifndef _CODE_LIBS_STRUCTNETLIB_ORIENTEDPATTERN_INCLUDE_GUARD
#define _CODE_LIBS_STRUCTNETLIB_ORIENTEDPATTERN_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>
#include <algorithm>
#include "../NetLib/NetLib.h"
#include "../UtilLib/UtilLib.h"

using UtilLib::Number;
using NetLib::Pattern;
using NetLib::PatternParsingException;
using std::ostream;


namespace StructnetLib
{
	//! OrientedPattern
	//! An OrientedPatterns is a Pattern which has x, y and feature dimensions.
	//! It is typically used to model a single layer in biologically motivated
	//! neural networks

	template <class PatternValue>
	class OrientedPattern : public Pattern<PatternValue> { 
	public:

		typedef			PatternValue& pattern_ref;
		typedef const	PatternValue& const_pattern_ref;
	 
		//! Default constructor (necessary for vector operations)
		OrientedPattern():_nr_x(0),_nr_y(0),_nr_o(0){}

		//! Standard constructor: specify the numver of pixels in the x direction,
		//! y-direction and the number of features in each layer
		OrientedPattern
		(	
			Number n_x, 
			Number n_y, 
			Number n_feature 
		);		

		//! Read an OrientedPattern from the input stream
		OrientedPattern
		( 
			istream& 
		);

		//! copy constructor
		OrientedPattern
		(
			const OrientedPattern<PatternValue>&
		);

		OrientedPattern&
			operator=
			(
				const OrientedPattern<PatternValue>&
			);

		//! Refer to a PatternValue by means of its spatial position
		PatternValue& operator()
		(
			Index, 
			Index, 
			Index
		);

		//! const version of previous function
		const   PatternValue&  operator()(Index, Index, Index) const;

		//! XML-Tag for serialization
		virtual string Tag        () const;

		//! Write to stream
		virtual bool   ToStream  (ostream&) const;

		//!Read from stream
		virtual bool   FromStream(istream&);

		//! standard streaming function
		template <class T> friend ostream& operator<<(ostream&, const OrientedPattern<T>& );

		//! Number of 'pixels' in x-direction
		Number				NrX()  const { return _nr_x; }

		//! Number of 'pixels' in y-direction
		Number				NrY()  const { return _nr_x; }

		//! Number of features (or 'orientations')
		Number				NrOr() const { return _nr_o; }

		//! Convert spatial position to an Index (the base class is the Pattern class, which supports operator[])
		Index				IndexFunction(Index, Index, Index) const;

		//! Get the x-position from the index value
		Number				NXFromIndex (Index) const;

		//! Get the y-position from the index value
		Number				NYFromIndex (Index) const;

		//! Get the feature (or orientation) from the index value
		Number				NORFromIndex(Index) const;

		//! A translation is possible if the whole figure can be shifted left, right, up 
		//! and down without being mutilated
		//! MaxTrx returns true if translations to the right are possible at all and inserts a positive
		//! value (indicating shifts to the right).
		bool				MaxTrx( int* ) const;

		//! MaxTry returns true if downward translations  are possible at all and inserts a positive 
		//! value (indicating shifts downward).
		bool				MaxTry( int* ) const;

		//! MinTrx returns true if translations to the left are possible at all and inserts a NEGATIVE
		//! value (indicating shifts to the left).
		bool				MinTrx( int* ) const;

		//! MinTry returns true if translations above are possible at all and inserts a NEGATIVE
		//! value (indicating shifts to upward).
		bool				MinTry( int* ) const;

		//! Carry out a translation in the x-direction, returns true if the translation can be carried out
		//! the pattern will contain the translated version of the original pattern; returns false if the
		//! translation can not be carried out. Patterns remains unchanged.
		bool				TransX(int);

		//! Carry out a translation in the y-direction, returns true if the translation can be carried out
		//! the pattern will contain the translated version of the original pattern; returns false if the
		//! translation can not be carried out. Patterns remains unchanged.
		bool				TransY(int);


	private:

		Number	CountNonZero() const;

		Number	NumberOfElements
				(
					Index, 
					Index, 
					Index
				) const;

		int		_nr_x; // physical dimension of this pattern: number of pixels in x-direction
		int		_nr_y; // physical dimension of this pattern: number of pixels in y-direction
		int		_nr_o; // physical dimension of this pattern: number of features (orientations)


	}; // end of OrientedPattern

	typedef OrientedPattern<double> D_OrientedPattern;

	template <class PatternValue>
	bool operator==( const OrientedPattern<PatternValue>&, const OrientedPattern<PatternValue>& );
} // end of Strucnet


#endif
