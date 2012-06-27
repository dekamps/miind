/*
 * WilsonCowanAlgorithm.cpp
 *
 *  Created on: 07.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/algorithm/RateAlgorithm.hpp>

namespace MPILib {
namespace algorithm{


RateAlgorithm::RateAlgorithm(Rate rate) :
		AlgorithmInterface<double>(), _time_current(
				std::numeric_limits<double>::max()), _rate(rate) {
}

RateAlgorithm::~RateAlgorithm() {
}

RateAlgorithm* RateAlgorithm::clone() const {
	return new RateAlgorithm(*this);
}

void RateAlgorithm::configure(
		const SimulationRunParameter& simParam) {

	_time_current = simParam.getTBegin();

}

void RateAlgorithm::evolveNodeState(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector, Time time) {
	nodeVector[0];
	weightVector[0];
	_time_current = time;
}

Time RateAlgorithm::getCurrentTime() const {
	return _time_current;

}

Rate RateAlgorithm::getCurrentRate() const {
	return _rate;
}

AlgorithmGrid RateAlgorithm::getGrid() const {
	std::vector<double> vector_grid(RATE_STATE_DIMENSION, _rate);
	std::vector<double> vector_interpretation(RATE_STATE_DIMENSION, 0);
	return AlgorithmGrid(vector_grid, vector_interpretation);
}

} /* namespace algorithm */
} /* namespace MPILib */
