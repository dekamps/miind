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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_POPULISTLIB_CHARACTERISTICALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_CHARACTERISTICALGORITHMCODE_INCLUDE_GUARD


#include "CharacteristicAlgorithm.h"

namespace PopulistLib {

	template <class Weight>
	CharacteristicAlgorithm<Weight>::CharacteristicAlgorithm
	(
		const CharacteristicParameter& par_char
	):
	AbstractAlgorithm<Weight>(0),
	_t_current(0),
	_rate_current(0),
	_grid(0),
	_state(vector<Weight>(0))
	{
	}

	template <class Weight>
	CharacteristicAlgorithm<Weight>::CharacteristicAlgorithm
	(
		const CharacteristicAlgorithm<Weight>& rhs
	):AbstractAlgorithm<Weight>(rhs),
	_t_current		(rhs._t_current),
	_rate_current	(rhs._rate_current),
	_grid			(rhs._grid),
	_state			(rhs._state)
	{
	}

	template <class Weight>
	bool CharacteristicAlgorithm<Weight>::Configure(const SimulationRunParameter& par_run)
	{
		return true;
	}

	template <class Weight>
	CharacteristicAlgorithm<Weight>::~CharacteristicAlgorithm()
	{
	}

	template <class Weight>
	Time CharacteristicAlgorithm<Weight>::CurrentTime() const
	{
		return _t_current;
	}

	template <class Weight>
	Rate CharacteristicAlgorithm<Weight>::CurrentRate() const
	{
		return _rate_current;
	}

	template <class Weight>
	AlgorithmGrid CharacteristicAlgorithm<Weight>::Grid() const
	{
		return _grid;
	}

	template <class Weight>
	bool CharacteristicAlgorithm<Weight>::Dump(ostream&) const
	{
		return true;
	}

	template <class Weight>
	bool CharacteristicAlgorithm<Weight>::EvolveNodeState
	(
		predecessor_iterator iter_begin,
		predecessor_iterator iter_end,
		Time time
	)
	{
		return true;
	}

	template <class Weight>
	NodeState CharacteristicAlgorithm<Weight>::State() const
	{
		return _state;
	}

	template <class Weight>
	string CharacteristicAlgorithm<Weight>::LogString() const
	{
		return "";
	}

	template <class Weight>
	CharacteristicAlgorithm<Weight>* CharacteristicAlgorithm<Weight>::Clone() const
	{
		return new CharacteristicAlgorithm<Weight>(*this);
	}
}

#endif // include guard
