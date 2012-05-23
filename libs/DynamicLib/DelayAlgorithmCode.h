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
#ifndef _CODE_LIBS_DYNAMICLIB_DELAYALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_DELAYALGORITHMCODE_INCLUDE_GUARD

#include <boost/lexical_cast.hpp>
#include "DelayAlgorithm.h"

namespace DynamicLib {
	template <class WeightValue>
	DelayAlgorithm<WeightValue>::DelayAlgorithm(istream& s):
	_t_current(0.0),
	_t_delay(0.0),
	_rate_current(0.0)
	{
		this->FromStream(s);
	}

	template <class WeightValue>
	DelayAlgorithm<WeightValue>::DelayAlgorithm(Time t_delay):
	_t_current(0.0),
	_t_delay(t_delay),
	_rate_current(0.0)
	{
	}

	template <class WeightValue>
	DelayAlgorithm<WeightValue>::~DelayAlgorithm()
	{
	}


	template <class WeightValue>
	DelayAlgorithm<WeightValue>* DelayAlgorithm<WeightValue>::Clone() const
	{
		return new DelayAlgorithm(*this);
	}

	template <class WeightValue>
	NodeState DelayAlgorithm<WeightValue>::State() const
	{
		vector<double> state(1);
		state[0] = this->CurrentRate();
		return state;
	}

	template <class WeightValue>
	AlgorithmGrid DelayAlgorithm<WeightValue>::Grid() const
	{
		AlgorithmGrid grid(1);
		return grid;
	}

	template <class WeightValue>
	bool DelayAlgorithm<WeightValue>::Dump(ostream& s) const
	{
		return false;
	}

	template <class WeightValue>
	string DelayAlgorithm<WeightValue>::LogString() const
	{
		return "";
	}

	template <class WeightValue>
	Rate DelayAlgorithm<WeightValue>::CurrentRate() const
	{
		return _rate_current;
	}

	template <class WeightValue>
	Time DelayAlgorithm<WeightValue>::CurrentTime() const
	{
		return _t_current;
	}

	template <class WeightValue>
	bool DelayAlgorithm<WeightValue>::Configure(const SimulationRunParameter& par)
	{
		_t_current = par.TBegin();
		return true;
	}

	template <class WeightValue>
	bool DelayAlgorithm<WeightValue>::EvolveNodeState
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end,
		Time t
	)
	{

		Rate rate = DecodeCurrentInputRate(iter_begin,iter_end);

		
		rate_time_pair p;
		p.first = rate;

		if (_queue.size() == 0){
			p.second = _t_delay;
			_queue.push_back(p);
		}

		_t_current = t;
		p.second = _t_current + _t_delay;
		_queue.push_back(p);
		_rate_current = CalculateDelayedRate();

		return true;
	}

	template <class WeightValue>
	Rate DelayAlgorithm<WeightValue>::CalculateDelayedRate() 
	{
		int i = 0;
		while(i < static_cast<int>(_queue.size()) && _queue[i].second <= _t_current)
			i++;
		if (i == 0 )
			return 0.0;		

		for (int j = 0; j < i-1; j++)
			_queue.pop_front();

		return this->Interpolate();

	}

	template <class WeightValue>
	Rate DelayAlgorithm<WeightValue>::DecodeCurrentInputRate
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end
	) const
	{
		Rate rate = iter_begin->GetValue();
		return rate;
	}

	template <class WeightValue>
	string DelayAlgorithm<WeightValue>::Tag() const 
	{
		return "<DelayAlgorithm>";
	}

	template <class WeightValue>
	bool DelayAlgorithm<WeightValue>::ToStream(ostream& s) const
	{
		s << Tag() << _t_delay << this->ToEndTag(this->Tag()) << "\n";
		return true;
	}

	template <class WeightValue>
	Rate DelayAlgorithm<WeightValue>::Interpolate() const {

		double t_early  = _queue[0].second;
		double t_late   = _queue[1].second;
		assert(t_late >= _t_current && t_early <= _t_current);
		double t_dif   = t_late - t_early;
		double t_rat   = _t_current - t_early;
		double alpha = t_rat / t_dif;
		Rate rate = alpha*_queue[0].first + (1-alpha)*_queue[1].first;

		return rate;
	}

	template <class WeightValue>
	bool DelayAlgorithm<WeightValue>::FromStream(istream& s)
	{
		string dummy;
		s >> dummy;
		string float_value = this->UnWrapTag(dummy);
		_t_delay = boost::lexical_cast<double>(float_value);
		return true;
	}
}

#endif // include guard
