// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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

#ifndef MPILIB_ALGORITHMS_DELAYALGORITHM_CODE_HPP_
#define MPILIB_ALGORITHMS_DELAYALGORITHM_CODE_HPP_

//#include <MPILib/config.hpp>
#include <cassert>
#include <MPILib/algorithm/DelayAlgorithm.hpp>

namespace MPILib {
namespace algorithm {

template<class Weight>
DelayAlgorithm<Weight>::DelayAlgorithm(Time t_delay) :
		algorithm::AlgorithmInterface<Weight>(), _t_current(0.0), _t_delay(
				t_delay), _rate_current(0.0) {
}

template<class Weight>
DelayAlgorithm<Weight>::~DelayAlgorithm() {
}

template<class Weight>
DelayAlgorithm<Weight>* DelayAlgorithm<Weight>::clone() const {
	return new DelayAlgorithm(*this);
}
template<class Weight>
void DelayAlgorithm<Weight>::configure(const SimulationRunParameter& par) {
	_t_current = par.getTBegin();
}

template<class Weight>
void DelayAlgorithm<Weight>::evolveNodeState(
		const std::vector<Rate>& nodeVector,
		const std::vector<Weight>& weightVector, Time time) {

        assert(nodeVector.size() == 1);
	assert(weightVector.size() == 1);
	Rate rate = (*nodeVector.begin());

	rate_time_pair p;
	p.first = rate;

	if (_queue.size() == 0) {
		p.second = _t_delay;
		_queue.push_back(p);
	}

	_t_current = time;
	p.second = _t_current + _t_delay;
	_queue.push_back(p);
	_rate_current = CalculateDelayedRate();

}

template<class Weight>
Time DelayAlgorithm<Weight>::getCurrentTime() const {
	return _t_current;
}

template<class Weight>
Rate DelayAlgorithm<Weight>::getCurrentRate() const {
        assert(_rate_current >= 0.0);
	return _rate_current;
}

template<class Weight>
AlgorithmGrid DelayAlgorithm<Weight>::getGrid(MPILib::NodeId) const {
	AlgorithmGrid grid(1);
	return grid;
}

template<class Weight>
Rate DelayAlgorithm<Weight>::CalculateDelayedRate() {
	int i = 0;
	while (i < static_cast<int>(_queue.size()) && _queue[i].second <= _t_current)
		i++;
	if (i == 0)
		return 0.0;

	for (int j = 0; j < i - 1; j++)
		_queue.pop_front();

	return this->Interpolate();
}

template<class Weight>
Rate DelayAlgorithm<Weight>::Interpolate() const {

	if (_queue.size() == 1){
		// this happens if the delay is 0
		assert(_t_current == _queue[0].second);
		return _queue[0].first;
	}
	double t_early  = _queue[0].second;
	double t_late   = _queue[1].second;
	assert(t_late >= _t_current && t_early <= _t_current);
	double t_dif   = t_late - t_early;
	double t_rat   = _t_current - t_early;
	double alpha = t_rat / t_dif;
	Rate rate = alpha*_queue[0].first + (1-alpha)*_queue[1].first;

	return rate;

}

} /* end namespace algorithm */
} /* end namespace MPILib */

#endif  //include guard MPILIB_ALGORITHMS_DELAYALGORITHM_CODE_HPP_
