/*
 * WilsonCowanAlgorithm.cpp
 *
 *  Created on: 07.06.2012
 *      Author: david
 */

#ifndef MPILIB_ALGORITHMS_RATEALGORITHM_CODE_HPP_
#define MPILIB_ALGORITHMS_RATEALGORITHM_CODE_HPP_

#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/algorithm/RateAlgorithm.hpp>

namespace MPILib {
namespace algorithm{

template<class Weight>
RateAlgorithm<Weight>::RateAlgorithm(Rate rate) :
		AlgorithmInterface<double>(), _time_current(
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
AlgorithmGrid RateAlgorithm<Weight>::getGrid() const {
	std::vector<double> vector_grid(RATE_STATE_DIMENSION, _rate);
	std::vector<double> vector_interpretation(RATE_STATE_DIMENSION, 0);
	return AlgorithmGrid(vector_grid, vector_interpretation);
}

} /* namespace algorithm */
} /* namespace MPILib */

#endif //end include guard MPILIB_ALGORITHMS_RATEALGORITHM_CODE_HPP_


