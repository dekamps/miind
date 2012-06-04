/*
 * AlgorithmInterface.hpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */

#ifndef MPILIB_ALGORITHMINTERFACE_HPP_
#define MPILIB_ALGORITHMINTERFACE_HPP_

#include <MPILib/include/BasicTypes.hpp>
#include <vector>


namespace MPILib {

class AlgorithmInterface {
public:
	AlgorithmInterface(){};
	virtual ~AlgorithmInterface(){};

	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual AlgorithmInterface* Clone() const = 0;

	/**
	 * Configure the Algorithm
	 * @param simParam
	 */
	virtual void Configure(const SimulationRunParameter& simParam) = 0;

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void EvolveNodeState(const std::vector<NodeState>& nodeVector,
			const std::vector<WeightType>& weightVector, Time time) = 0;
};

} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMINTERFACE_HPP_ */
