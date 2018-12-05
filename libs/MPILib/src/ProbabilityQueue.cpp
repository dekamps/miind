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
#include <MPILib/include/ProbabilityQueue.hpp>
#include <iostream>

using namespace MPILib;
using namespace MPILib::populist;

ProbabilityQueue::ProbabilityQueue
(
	Time t_batch_size
):
_scale				(1.0),
_t_batch_size		(t_batch_size),
_t_current			(0),
_t_current_batch	(0),
_prob_current_batch	(0),
_total				(0)
{
}

ProbabilityQueue::ProbabilityQueue(const ProbabilityQueue& rhs):
_scale(rhs._scale),
_t_batch_size(rhs._t_batch_size),
_t_current(rhs._t_current),
_t_current_batch(rhs._t_current_batch),
_prob_current_batch(rhs._prob_current_batch),
_total(rhs._total),
_queue(rhs._queue)
{
}

void ProbabilityQueue::push(const StampedProbability& prob)
{
	_t_current = prob._time;

	Number n_steps = static_cast<Number>(prob._time/_t_batch_size);
	Time t_this_batch = n_steps*_t_batch_size;

	if ( t_this_batch > _t_current_batch ){
		StampedProbability pqueue;
		pqueue._prob  = _prob_current_batch;
		pqueue._time  = _t_current_batch;
		_queue.push(pqueue);
		_t_current_batch = t_this_batch;
		_prob_current_batch = 0.0;
	} 
	_prob_current_batch += prob._prob;
	_total += prob._prob;
}

Probability ProbabilityQueue::CollectAndRemove(Time time)
{
	Probability p = 0;	

	Number n_steps = static_cast<Number>(time/_t_batch_size);
	Time t_this_batch = n_steps*_t_batch_size;
	
	if (t_this_batch > _t_current_batch){
		_t_current_batch = t_this_batch;
		p += _prob_current_batch;
		_prob_current_batch = 0.0;
	}

	while (_queue.size() && _queue.front()._time <=  time){
		p += _queue.front()._prob;
		_queue.pop();
	}
	_total -= p;
	return _scale*p;
}

bool ProbabilityQueue::HasProbability(Time time) const
{
	if (! _queue.size() )
		return false;
	return (time <= _queue.front()._time )  ? true: false;
}

void ProbabilityQueue::Scale(double scale)
{
	_scale *= scale;
}
