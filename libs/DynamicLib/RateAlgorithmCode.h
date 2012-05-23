// Copyright (c) 2005 - 2011 Marc de Kamps
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

#ifndef _CODE_LIBS_DYNAMICLIB_RATEALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_RATEALGORITHMCODE_INCLUDE_GUARD

#include "RateAlgorithm.h"

namespace DynamicLib {

	template <class Weight>
	RateAlgorithm<Weight>::RateAlgorithm(istream& s):
	AbstractAlgorithm<Weight>(RATE_STATE_DIMENSION),
	_time_current(0),
	_p_rate(0)			// fixed: 08/08/11: if not set to zero it may trigger a dereference in Clone
	{
		string dummy;

		s >> dummy;

		if ( this->IsAbstractAlgorithmTag(dummy) ){
			s >> dummy;
			s >> dummy;
		}

		ostringstream str;
		str << dummy << ">";
		if ( str.str() != this->Tag() )
			throw DynamicLibException("RateAlgorithm tag expected");

		getline(s,dummy);

		string name_alg;
		if (! this->StripNameFromTag(&name_alg, dummy) )
				throw DynamicLibException("RateAlgorithm tag expected");
		this->SetName(name_alg);

		s >> _rate;

		this->StripFooter(s);
	}

	template <class Weight>
	void RateAlgorithm<Weight>::StripFooter(istream& s)
	{
		string dummy;

		s >> dummy;

		if ( dummy != this->ToEndTag(this->Tag() ) )
			throw DynamicLibException("RateAlgorithm end tag expected");

		// absorb the AbstractAlgorithm tag
		s >> dummy;

	}


	template <class Weight>
	RateAlgorithm<Weight>::RateAlgorithm
	(
		Rate rate
	):
	AbstractAlgorithm<Weight>(RATE_STATE_DIMENSION),
	_time_current(numeric_limits<double>::max()),
	_rate(rate),
	_p_rate(0)
	{
	}

	template <class Weight>
	RateAlgorithm<Weight>::RateAlgorithm(const RateAlgorithm<Weight>& rhs):
	AbstractAlgorithm<Weight>(rhs),
	_time_current(rhs._time_current),
	_rate(rhs._rate),
	_p_rate(rhs._p_rate)
	{
	}

	template <class Weight>
	RateAlgorithm<Weight>::RateAlgorithm
	(
		Rate* p_rate
	):
	AbstractAlgorithm<Weight>(RATE_STATE_DIMENSION),
	_time_current(numeric_limits<double>::max()),
	_rate(0),
	_p_rate(p_rate)
	{
	}

	template <class Weight>
	RateAlgorithm<Weight>::~RateAlgorithm()
	{
	}

	template <class Weight>
	NodeState RateAlgorithm<Weight>::State() const
	{
		vector<double> vector_state(1,_rate);
		return NodeState(vector_state);
	}

	template <class Weight>
	RateAlgorithm<Weight>* RateAlgorithm<Weight>::Clone() const
	{
		return new RateAlgorithm(*this);
	}

	template <class Weight>
	string	RateAlgorithm<Weight>::LogString() const
	{
		return string("");
	}

	template <class Weight>
	bool RateAlgorithm<Weight>::EvolveNodeState
	(
		predecessor_iterator, 
		predecessor_iterator,
		Time time_to_achieve
	)
	{
		_time_current = time_to_achieve;
		return true;
	}

	template <class Weight>
	AlgorithmGrid RateAlgorithm<Weight>::Grid() const
	{
		vector<double> vector_grid           (RATE_STATE_DIMENSION,_rate);
		vector<double> vector_interpretation (RATE_STATE_DIMENSION,0    );
		return AlgorithmGrid(vector_grid, vector_interpretation);
	}

	template <class Weight>
	bool RateAlgorithm<Weight>::Dump(ostream& s) const
	{
		return true;
	}

	template <class Weight>
	bool RateAlgorithm<Weight>::Configure		
	(
		const SimulationRunParameter& parameter_simulation
	)
	{
		_time_current = parameter_simulation.TBegin();

		return true;
	}

	template <class Weight>
	Time RateAlgorithm<Weight>::CurrentTime() const
	{
		return _time_current;
	}

	template <class Weight>
	Rate RateAlgorithm<Weight>::CurrentRate() const
	{
		return (_p_rate) ? *_p_rate : _rate;
	}

	template <class Weight>
	bool RateAlgorithm<Weight>::FromStream(istream& s)
	{
		return true;
	}

	template <class Weight>
	bool RateAlgorithm<Weight>::ToStream(ostream& s) const
	{
		this->AbstractAlgorithm<Weight>::ApplyBaseClassHeader(s,"RateAlgorithm");

		s << this->InsertNameInTag(this->Tag(),this->GetName())	<< "\n";
		s << CurrentRate()	<< "\n";
		s << this->ToEndTag(Tag())<< "\n";

		this->AbstractAlgorithm<Weight>::ApplyBaseClassFooter(s);

		return true;
	}

	template <class Weight>
	string RateAlgorithm<Weight>::Tag() const
	{
		return STR_RA_TAG;
	}


} // end of DynamicLib

#endif // include guard
