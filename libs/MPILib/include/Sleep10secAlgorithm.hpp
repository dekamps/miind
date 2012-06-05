/*
 * Sleep10secAlgorithm.hpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */

#ifndef MPILIB_SLEEP10SECALGORITHM_HPP_
#define MPILIB_SLEEP10SECALGORITHM_HPP_

#include <MPILib/include/AlgorithmInterface.hpp>
#include <vector>
#include <MPILib/include/BasicTypes.hpp>

namespace MPILib {

template <class WeightValue>
class Sleep10secAlgorithm: public MPILib::AlgorithmInterface<WeightValue> {
public:
	explicit Sleep10secAlgorithm();

	virtual ~Sleep10secAlgorithm();
	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual Sleep10secAlgorithm* Clone() const;

	/**
	 * Configure the Algorithm
	 * @param simParam
	 */
	virtual void Configure(const SimulationRunParameter& simParam);

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void EvolveNodeState(const std::vector<NodeState>& nodeVector,
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
};

} /* namespace MPILib */
#endif /* MPILIB_SLEEP10SECALGORITHM_HPP_ */
