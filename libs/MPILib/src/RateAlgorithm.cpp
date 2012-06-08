/*
 * WilsonCowanAlgorithm.cpp
 *
 *  Created on: 07.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/RateAlgorithm.hpp>

namespace MPILib {

RateAlgorithm::RateAlgorithm(Rate* rate) :
		AlgorithmInterface<double>(), _time_current(
				numeric_limits<double>::max()), _rate(0), _p_rate(rate) {
}

RateAlgorithm::RateAlgorithm(Rate rate) :
		AlgorithmInterface<double>(), _time_current(
				numeric_limits<double>::max()), _rate(rate), _p_rate(0) {
}

RateAlgorithm::~RateAlgorithm() {
	// TODO Auto-generated destructor stub
}

RateAlgorithm* RateAlgorithm::Clone() const {
	return new RateAlgorithm(*this);
}

void RateAlgorithm::Configure(
		const DynamicLib::SimulationRunParameter& simParam) {

	_time_current = simParam.TBegin();

}

void RateAlgorithm::EvolveNodeState(const std::vector<Rate>& nodeVector,
		const std::vector<double>& weightVector, Time time) {

	_time_current = time;
}

Time RateAlgorithm::getCurrentTime() const {
	return _time_current;

}

Rate RateAlgorithm::getCurrentRate() const {
	return (_p_rate) ? *_p_rate : _rate;
}

DynamicLib::AlgorithmGrid RateAlgorithm::Grid() const {
	std::vector<double> vector_grid(DynamicLib::RATE_STATE_DIMENSION, _rate);
	std::vector<double> vector_interpretation(DynamicLib::RATE_STATE_DIMENSION, 0);
	return DynamicLib::AlgorithmGrid(vector_grid, vector_interpretation);
}

} /* namespace MPILib */
