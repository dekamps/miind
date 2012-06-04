/*
 * EmptyAlgorithm.h
 *
 *  Created on: 04.06.2012
 *      Author: david
 */

#ifndef MPILIB_EMPTYALGORITHM_H_
#define MPILIB_EMPTYALGORITHM_H_

#include <MPILib/include/AlgorithmInterface.hpp>
#include <vector>
#include <MPILib/include/BasicTypes.hpp>

namespace MPILib {

class EmptyAlgorithm: public MPILib::AlgorithmInterface {
public:
	explicit EmptyAlgorithm();

	virtual ~EmptyAlgorithm();
	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual EmptyAlgorithm* Clone() const;

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
			const std::vector<WeightType>& weightVector, Time time);
};

} /* namespace MPILib */
#endif /* MPILIB_EMPTYALGORITHM_H_ */
