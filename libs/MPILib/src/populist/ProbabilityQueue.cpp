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
#include <MPILib/include/populist/ProbabilityQueue.hpp>
#include <MPILib/include/utilities/Exception.hpp>

#include <cassert>
#include <math.h>

namespace MPILib {
namespace populist {

void ProbabilityQueue::push(const StampedProbability& prob)
{
	assert(prob._time >= _current._time); // accept 0 == 0
	if (prob._time < _time_current)
		throw utilities::Exception("Pushed an old event on queue.");

	_total +=  prob._prob;
	if ( prob._time -_current._time > _time_step){
		_current._time = floor(prob._time/_time_step)*_time_step;
		_queue.push(_current);
		_current._prob = 0.;
	}
	_current._prob += prob._prob;

}

Probability ProbabilityQueue::CollectAndRemove(Time time)
{
	_time_current = floor(time/_time_step)*_time_step;
	Probability total = 0;
	while ( !_queue.empty() && _queue.front()._time <= time )
	{
		total += _queue.front()._prob;
		_queue.pop();
	}
	
	_total -= total;
	return total;
}

bool ProbabilityQueue::HasProbability(Time time) const
{
	if (_queue.empty() )
		return false;
	else
		if (_queue.front()._time <= time )
			return true;
		else
			return false;
}

void ProbabilityQueue::Scale(double scale){
	_current._prob *= scale;
	_total *= scale;

	_queue.Scale(scale);
}

} /* namespace populist */
} /* namespace MPILib */
