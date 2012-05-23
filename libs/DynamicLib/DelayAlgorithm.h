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
#ifndef _CODE_LIBS_DYNAMICLIB_DELAYALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_DELAYALGORITHM_INCLUDE_GUARD

#include <deque>
#include <iostream>
#include "AbstractAlgorithm.h"

using std::deque;
using std::istream;

namespace DynamicLib {

	//! This algorithm is effectively a pipeline with a preselected delay.

	//! In some simulations connections must be implemented with time delays. If that needs to be done with
	//! high precision, create a node, configure it with a DelayAlgorithm, connected the output to be delayed
	//! to this node and connect the output of this node to the node specified by the original connection. At the
	//! moment this is the only way to implement delays. A less precise effect can be achieved with Wilson-Cowan algorithms.
	//! For large-scale simulations this solution may not be sustainable. Please provide feedback if this is the case.

	template <class WeightValue>
	class DelayAlgorithm : public AbstractAlgorithm<WeightValue> {
	public:

		typedef typename AbstractAlgorithm<WeightValue>::predecessor_iterator predecessor_iterator;

		//! Construct an algorithm from a stream
		DelayAlgorithm(istream&);

		//! Create algorithm with a delay time
		DelayAlgorithm(Time);

		//! destructor
		virtual ~DelayAlgorithm();

		//! essentially noop
		NodeState State() const;

		//! essentially noop
		AlgorithmGrid Grid() const;

		//! configure simulation
		bool Configure(const SimulationRunParameter&);

		//! clone algorithm, responsibility for destruction lies with client
		virtual DelayAlgorithm<WeightValue>* Clone() const;

		//! current time according to algorithm
		Time CurrentTime() const;

		//! current rate of algorithm
		Rate CurrentRate() const;

		//!
		string LogString() const;

		//!  essentially noop
		bool Dump(ostream&) const;

		//! DelayAlgorithm accepts only a single input node. It will store the current input an a queue, and return as a rate,
		//! the input it received earlier.
		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			Time
		);

		//! define XML tag for DelayAlgorithm
		virtual string Tag() const;

		//! stream algorithm to an output stream
		virtual bool ToStream(ostream&) const;

		//! stream algorithm from an input stream
		virtual bool FromStream(istream&);
	private: 

		Rate DecodeCurrentInputRate
		(
			predecessor_iterator,
			predecessor_iterator
		) const;

		Rate CalculateDelayedRate(); 

		Rate Interpolate() const;

		typedef pair<Rate, Time> rate_time_pair;

		Time	_t_current;
		Time	_t_delay;
		Rate	_rate_current;

		deque<rate_time_pair> _queue;

	};
} // namespace

#endif  //include guard
