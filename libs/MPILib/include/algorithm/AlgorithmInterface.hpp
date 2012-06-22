/*
 * AlgorithmInterface.hpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */

#ifndef MPILIB_ALGORITHMS_ALGORITHMINTERFACE_HPP_
#define MPILIB_ALGORITHMS_ALGORITHMINTERFACE_HPP_

#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/algorithm/AlgorithmGrid.hpp>
#include <vector>


namespace MPILib {
namespace algorithm{

template <class WeightValue>
class AlgorithmInterface {
public:
	AlgorithmInterface(){};
	virtual ~AlgorithmInterface(){};

	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual AlgorithmInterface* clone() const = 0;

	/**
	 * Configure the Algorithm
	 * @param simParam
	 */
	virtual void configure(const SimulationRunParameter& simParam) = 0;

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<WeightValue>& weightVector, Time time) = 0;

	/**
	 * The current timepoint
	 * @return The current time point
	 */
	virtual Time getCurrentTime() const = 0;

	/**
	 * The calculated rate of the node
	 * @return The rate of the node
	 */
	virtual Rate getCurrentRate() const = 0;

	/**
	 * Stores the algorithm state in a Algorithm Grid
	 * @return The state of the algorithm
	 */
	virtual AlgorithmGrid getGrid() const = 0;
};

} /* namespace algorithm */
} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMS_ALGORITHMINTERFACE_HPP_ */
