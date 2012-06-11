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
				numeric_limits<double>::max()), _rate(rate) {
}

RateAlgorithm::~RateAlgorithm() {
	// TODO Auto-generated destructor stub
}

RateAlgorithm* RateAlgorithm::clone() const {
	return new RateAlgorithm(*this);
}

void RateAlgorithm::configure(
		const DynamicLib::SimulationRunParameter& simParam) {

	_time_current = simParam.TBegin();

}

void RateAlgorithm::evolveNodeState(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector, Time time) {

	_time_current = time;
}

Time RateAlgorithm::getCurrentTime() const {
	return _time_current;

}

Rate RateAlgorithm::getCurrentRate() const {
	return _rate;
}

DynamicLib::AlgorithmGrid RateAlgorithm::getGrid() const {
	std::vector<double> vector_grid(DynamicLib::RATE_STATE_DIMENSION, _rate);
	std::vector<double> vector_interpretation(DynamicLib::RATE_STATE_DIMENSION, 0);
	return DynamicLib::AlgorithmGrid(vector_grid, vector_interpretation);
}

} /* namespace algorithm */
} /* namespace MPILib */
