/*
 * Sleep10secAlgorithm.hpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */

#ifndef MPILIB_ALGORITHMS_SLEEPALGORITHM_HPP_
#define MPILIB_ALGORITHMS_SLEEPALGORITHM_HPP_

#include <vector>
#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include <DynamicLib/NodeState.h>

namespace MPILib {
namespace algorithm {

template<class WeightValue>
class SleepAlgorithm: public AlgorithmInterface<WeightValue> {
public:
	explicit SleepAlgorithm();

	virtual ~SleepAlgorithm();
	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual SleepAlgorithm* clone() const;

	/**
	 * Configure the Algorithm
	 * @param simParam
	 */
	virtual void configure(const DynamicLib::SimulationRunParameter& simParam);

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void evolveNodeState(const std::vector<ActivityType>& nodeVector,
			const std::vector<WeightValue>& weightVector, Time time);

	/**
	 * The current timepoint
	 * @return The current time point
	 */
	virtual Time getCurrentTime() const;

	/**
	 * The calculated rate of the node
	 * @return The rate of the node
	 */
	virtual Rate getCurrentRate() const;

	virtual DynamicLib::AlgorithmGrid getGrid() const;

private:

	static double const kSleepTime = 1;

};

} /* namespace algorithm */
} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMS_SLEEPALGORITHM_HPP_ */
