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

#ifndef MPILIB_ALGORITHMS_RATEALGORITHM_CODE_HPP_
#define MPILIB_ALGORITHMS_RATEALGORITHM_CODE_HPP_

#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/BasicDefinitions.hpp>
#include "RateAlgorithm.hpp"

namespace MPILib {

template<class Weight>
RateAlgorithm<Weight>::RateAlgorithm(Rate rate) :
		AlgorithmInterface<Weight>(), _time_current(
				std::numeric_limits<double>::max()), _rate(rate) {
}

template<class Weight>
RateAlgorithm<Weight>::~RateAlgorithm() {
}

template<class Weight>
RateAlgorithm<Weight>* RateAlgorithm<Weight>::clone() const {
	return new RateAlgorithm(*this);
}

template<class Weight>
void RateAlgorithm<Weight>::configure(
		const SimulationRunParameter& simParam) {

	_time_current = simParam.getTBegin();

}

template<class Weight>
void RateAlgorithm<Weight>::evolveNodeState(const std::vector<Rate>& nodeVector,
		const std::vector<Weight>& weightVector, Time time) {
	_time_current = time;
}

template<class Weight>
Time RateAlgorithm<Weight>::getCurrentTime() const {
	return _time_current;

}

template<class Weight>
Rate RateAlgorithm<Weight>::getCurrentRate() const {
	return _rate;
}

template<class Weight>
AlgorithmGrid RateAlgorithm<Weight>::getGrid(NodeId) const {
	std::vector<double> vector_grid(RATE_STATE_DIMENSION, _rate);
	std::vector<double> vector_interpretation(RATE_STATE_DIMENSION, 0);
	return AlgorithmGrid(vector_grid, vector_interpretation);
}

} /* namespace MPILib */

#endif //end include guard MPILIB_ALGORITHMS_RATEALGORITHM_CODE_HPP_


