/*
 * AlgorithmInterface.hpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */

#ifndef MPILIB_ALGORITHMINTERFACE_HPP_
#define MPILIB_ALGORITHMINTERFACE_HPP_

#include <MPILib/include/BasicTypes.hpp>
#include <DynamicLib/SimulationRunParameter.h>
#include <DynamicLib/AlgorithmGrid.h>
#include <vector>


namespace MPILib {

template <class WeightValue>
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
	virtual void Configure(const DynamicLib::SimulationRunParameter& simParam) = 0;

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void EvolveNodeState(const std::vector<Rate>& nodeVector,
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

	virtual DynamicLib::AlgorithmGrid Grid() const = 0;
};

} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMINTERFACE_HPP_ */
