/*
 * AlgorithmInterface.hpp
 *
 *  Created on: 04.06.2012
 *      Author: david
 */

#ifndef MPILIB_ALGORITHMS_ALGORITHMINTERFACE_HPP_
#define MPILIB_ALGORITHMS_ALGORITHMINTERFACE_HPP_

#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/algorithm/AlgorithmGrid.hpp>
#include <MPILib/include/NodeType.hpp>
#include <MPILib/include/utilities/Exception.hpp>
#include <vector>

namespace MPILib {
namespace algorithm {

template<class WeightValue>
class AlgorithmInterface {
public:
	AlgorithmInterface() {
	}
	;
	virtual ~AlgorithmInterface() {
	}
	;

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
	 * Evolve the node state. Overwrite this method if your algorithm does not
	 * need to know the NodeTypes.
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<WeightValue>& weightVector, Time time){
		throw utilities::Exception("You need to overwrite this method in your algorithm"
				" if you want to use it");
	}

	/**
	 * Evolve the node state. In the default case it simply calls envolveNodeState
	 * without the NodeTypes. However if an algorithm needs the nodeTypes
	 * of the precursors overwrite this function.
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 * @param weightVector Vector of the NodeTypes of the precursors
	 */
	virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<WeightValue>& weightVector, Time time,
			const std::vector<NodeType>& typeVector) {
		this->evolveNodeState(nodeVector, weightVector, time);
	}

	/**
	 * prepare the Evolve method
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param weightVector Vector of the NodeTypes of the precursors
	 */
	virtual void prepareEvolve(const std::vector<Rate>& nodeVector,
			const std::vector<WeightValue>& weightVector,
			const std::vector<NodeType>& typeVector){};

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


	std::valarray<double>& ArrayState(AlgorithmGrid& grid) const
	{
		return grid.getArrayState();
	}

	std::valarray<double>& ArrayInterpretation(AlgorithmGrid& grid) const
	{
		return grid.getArrayInterpretation();
	}

	Number& StateSize(AlgorithmGrid& grid) const
	{
		return grid.getStateSize();
	}

	Number StateSize(const AlgorithmGrid & grid) const
	{
		return grid.getStateSize();
	}

};

} /* namespace algorithm */
} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMS_ALGORITHMINTERFACE_HPP_ */
