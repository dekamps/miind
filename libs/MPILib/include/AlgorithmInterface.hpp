// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef MPILIB_ALGORITHMS_ALGORITHMINTERFACE_HPP_
#define MPILIB_ALGORITHMS_ALGORITHMINTERFACE_HPP_

//#include <MPILib/config.hpp> //TODO: this file is actually created in the build directory ... why? leave for now
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/AlgorithmGrid.hpp>
#include <MPILib/include/NodeType.hpp>
#include <MPILib/include/utilities/Exception.hpp>
#include <vector>

namespace MPILib {
namespace algorithm {

/**
 * @brief The interface for all algorithm classes.
 *
 * This class provides the interface to which all developed algorithms need to implement.
 */
template<class WeightValue>
class AlgorithmInterface {
public:

	typedef WeightValue WeightType;

	AlgorithmInterface()=default
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
	 * @param simParam The simulation parameter
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
			const std::vector<WeightValue>& weightVector, Time time) {
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
	 * @param typeVector Vector of the NodeTypes of the precursors
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
	 * @param typeVector Vector of the NodeTypes of the precursors
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
	 * @return The state of the algorithm. The Grid must at least contain one element; an empty grid will cause a crash.
	 */
	virtual AlgorithmGrid getGrid() const = 0;


	std::valarray<double>& getArrayState(AlgorithmGrid& grid) const
	{
		return grid.getArrayState();
	}

	std::valarray<double>& getArrayInterpretation(AlgorithmGrid& grid) const
	{
		return grid.getArrayInterpretation();
	}

	Number& getStateSize(AlgorithmGrid& grid) const
	{
		return grid.getStateSize();
	}

	Number getStateSize(const AlgorithmGrid & grid) const
	{
		return grid.getStateSize();
	}

};

} /* namespace algorithm */
} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMS_ALGORITHMINTERFACE_HPP_ */
