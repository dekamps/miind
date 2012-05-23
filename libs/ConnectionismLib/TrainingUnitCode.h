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
#ifndef _CODE_LIBS_CONNECTIONISMLIB_TRAININGUNITCODE_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONISMLIB_TRAININGUNITCODE_INCLUDE_GUARD

#include "ConnectionismLibException.h"
#include "TrainingUnit.h"

namespace ConnectionismLib {

	template <class PatternValue>
	TrainingUnit<PatternValue>::TrainingUnit():
	_pattern_in(0),
	_pattern_out(0)
	{
	}

	template <class PatternValue>
	TrainingUnit<PatternValue>::TrainingUnit
	(
		const Pattern<PatternValue>& pattern_in, 
		const Pattern<PatternValue>& pattern_out 
	):
	_pattern_in(pattern_in),
	_pattern_out(pattern_out)
	{
	} 

	template <class PatternValue>
	inline TrainingUnit<PatternValue>::TrainingUnit(const TrainingUnit<PatternValue>& unit):
	_pattern_in(unit._pattern_in),
	_pattern_out(unit._pattern_out)
	{
	}

	template <class PatternValue>
	inline TrainingUnit<PatternValue>& TrainingUnit<PatternValue>::operator=(const TrainingUnit& unit)
	{
		if (&unit == this)
			return *this;
		else
		{
			_pattern_in  = unit._pattern_in;
			_pattern_out = unit._pattern_out;

			return *this;
		}
	}

	template <class PatternValue>
	inline const Pattern<PatternValue>& TrainingUnit<PatternValue>::InPat() const 
	{
		return _pattern_in;
	}

	template <class PatternValue>
	inline const Pattern<PatternValue>& TrainingUnit<PatternValue>::OutPat() const 
	{
		return _pattern_out;
	}

	template <class PatternValue>
	string TrainingUnit<PatternValue>::Tag() const
	{
		return string("<TrainingUnit>");
	}

	template <class PatternValue>
	bool TrainingUnit<PatternValue>::FromStream(istream& s)
	{
		string str;
		s >> str;

		if ( str != this->Tag() )
			throw ConnectionismLibException("TrainingUnit tag expected");

		s >> _pattern_in;
		s >> _pattern_out;

		s >> str;
		if ( str != ToEndTag(this->Tag()) )
			throw ConnectionismLibException("TrainingUnitend  tag expected");

		return true;

	}
	template <class PatternValue>
	bool TrainingUnit<PatternValue>::ToStream(ostream& s) const
	{
		s << Tag()			<< "\n";
		s << this->InPat()  << "\n";
		s << this->OutPat() << "\n";
		s << ToEndTag(Tag())<< "\n";;

		return true;
	}
	template <class PatternValue>
	ostream& operator<<(ostream& s, const TrainingUnit<PatternValue>& tu)
	{

		tu.ToStream(s);

		return s;
	}

	template <class PatternValue>
	istream& operator>>(istream& s, TrainingUnit<PatternValue>& tu)
	{
		tu.FromStream(s);

		return s;
	}
}

#endif // include guard
