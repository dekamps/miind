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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_NETLIB_PATTERNCODE_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_PATTERNCODE_INCLUDE_GUARD

#include "Pattern.h"
#include "LocalDefinitions.h"

namespace NetLib
{

	template < class PatternValue>
	Pattern<PatternValue>::Pattern():
	_vector_of_pattern(0)
	{
	}

	template <class PatternValue>
	Pattern<PatternValue>::Pattern(Number n_pat_length):
	_vector_of_pattern(n_pat_length)
	{
	}

	template <class PatternValue>
	Pattern<PatternValue>::Pattern(const Pattern<PatternValue>& rhs)
	{
		_vector_of_pattern = rhs._vector_of_pattern;
	}

	template < class PatternValue >
	Pattern<PatternValue>::~Pattern()
	{
	}

	template <class PatternValue>
	Number Pattern<PatternValue>::Size() const
	{
		return static_cast<Number>(_vector_of_pattern.size());
	}

	template < class PatternValue >
	inline PatternValue& Pattern<PatternValue>::operator[](Index n_index)
	{
		assert( n_index >= 0 && n_index < _vector_of_pattern.size() );
		return _vector_of_pattern[n_index];
	}

	template < class PatternValue >
	inline const PatternValue& Pattern<PatternValue>::operator[](Index n_index) const
	{
		assert( n_index >= 0 && n_index < _vector_of_pattern.size() );
		return _vector_of_pattern[n_index];
	}


	template < class PatternValue >
	Pattern<PatternValue>& Pattern<PatternValue>::operator=(const Pattern<PatternValue>& rhs)
	{
		if ( this == & rhs )
			return *this;

		_vector_of_pattern  = rhs._vector_of_pattern;

		return *this;
	}
	
	template < class PatternValue >
	void Pattern<PatternValue>::Clear()
	{
		_vector_of_pattern.assign(_vector_of_pattern.size(),0);
	}


	template <class PatternValue>
	void Pattern<PatternValue>::ClipMin(PatternValue val)
	{
	  Number size = this->Size();
	  for (Index i = 0; i < size; i++ )
	    if ( _vector_of_pattern[i] < val )
	      _vector_of_pattern[i] = val;
	  //else
	  //  nothing

	}
	 
	template <class PatternValue>
	void Pattern<PatternValue>::ClipMax(PatternValue val)
	{

		size_t n_ind;
		size_t n_pat = this->Size();
		for ( n_ind = 0; n_ind < n_pat; n_ind++ )
			if ( _vector_of_pattern[n_ind] > val )
				_vector_of_pattern[n_ind] = val;
	//	    else
	//			leave untouched
	}

	template <class PatternValue>
	void  Pattern<PatternValue>::RandomizeBinary(RandomGenerator& generator)
	{
	  // use the globally defined random generator
	  UniformDistribution ran_uni(generator);

		for (Index n_pat_index = 0; n_pat_index < Size(); n_pat_index++ )
			if ( ran_uni.NextSampleValue() > 0.5 )
				_vector_of_pattern[n_pat_index] = 1;
			else
				_vector_of_pattern[n_pat_index] = -1;
	}

	template <class PatternValue>
	Pattern<PatternValue>& Pattern<PatternValue>::operator+=(const Pattern<PatternValue>& pattern)
	{
		assert( pattern.Size() == Size() );

		Number n_size = Size();
		for (Index n_ind = 0; n_ind < n_size; n_ind++ )
			_vector_of_pattern[n_ind] += pattern._vector_of_pattern[n_ind];
	
		return *this;
	} 

	template <class PatternValue>
	bool Pattern<PatternValue>::ToStream(ostream& s) const
	{
		// Tested: 6-12-1999
		// Author: Marc de Kamps

		s << Pattern<PatternValue>::Tag() << "\n";
		s << static_cast<unsigned int>(_vector_of_pattern.size()) << "\n";
		for (Index n_ind = 0; n_ind < _vector_of_pattern.size(); n_ind++ )
			s << _vector_of_pattern[n_ind] << "\t";

		s << "\n" << ToEndTag(Pattern<PatternValue>::Tag()) << "\n";

		return true;	
	}

	template <class PatternValue>
	ostream& operator<<(ostream& s, const Pattern<PatternValue>& pattern)
	{
		pattern.ToStream(s);
		return s;
	}

	template <class PatternValue>
	bool Pattern<PatternValue>::FromStream(istream& s)
	{
		// Tested: 6-12-1999
		// Author: Marc de Kamps

		_vector_of_pattern.clear();
		string str;

		// check for correct begin:
		s >> str; 

		if ( str != Pattern<PatternValue>::Tag() )
			throw PatternParsingException(STR_HEADER_EXCEPTION);

		Number n_size;	// size of the pattern
		s >> n_size;

		if ( ! s.good() )
			throw PatternParsingException(STR_INT_EXCEPTION);

		PatternValue f_inp = 0;	// pattern value to be read in

		// read in the actual pattern:
		for (Index n_ind = 0 ; n_ind < n_size; n_ind++ )
		{
			s >> f_inp;
			_vector_of_pattern.push_back(f_inp);

			if (!s.good())
				throw PatternParsingException(STR_PATTERN_VALUE_EXCEPTION);
		}

		// check for correct ending:

		if ( !s.good() )
			throw PatternParsingException(STR_PATTERN_VALUE_EXCEPTION);

		s >> str;
		if ( str != ToEndTag(Pattern<PatternValue>::Tag()) )
			throw PatternParsingException(STR_FOOTER_EXCEPTION);

		return true;
	}

	template <class PatternValue>
	istream& operator>>(istream& s, Pattern<PatternValue>& pattern)
	{
		pattern.FromStream(s);
		return s;
	}

	template <class PatternValue>
	string Pattern<PatternValue>::Tag() const
	{
		return STR_PATTERN_TAG;
	}

	template <class PatternValue>
	typename vector<PatternValue>::iterator Pattern<PatternValue>::begin()
	{
		return _vector_of_pattern.begin();
	}


	template <class PatternValue>
	typename vector<PatternValue>::iterator Pattern<PatternValue>::end() 
	{
		return _vector_of_pattern.end();
	}

	template <class PatternValue> Pattern<PatternValue> operator+
	(
		const Pattern<PatternValue>& pat_1,
		const Pattern<PatternValue>& pat_2
	)
	{
		Pattern<PatternValue> pat_ret = pat_1;
		pat_ret += pat_2;
		return pat_ret;
	}

} // end of NetLib

#endif // include guard
