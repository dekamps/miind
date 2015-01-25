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
#ifndef MPILIB_POPULIST_POPOULATIONALGORITHM_HPP_
#define MPILIB_POPULIST_POPOULATIONALGORITHM_HPP_
#include <MPILib/include/MPINode.hpp>
#include <MPILib/include/MPINetwork.hpp>
#include <MPILib/include/DelayedConnection.hpp>

#include <MPILib/include/algorithm/AlgorithmInterface.hpp>

#include <MPILib/include/populist/PopulationGridControllerCode.hpp>
#include <MPILib/include/populist/parameters/PopulistParameter.hpp>
#include <MPILib/include/populist/RateFunctorCode.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/algorithm/RateAlgorithmCode.hpp>
#include <MPILib/include/algorithm/AlgorithmGrid.hpp>
#include <sstream>

namespace MPILib {
namespace populist {

class PopulistSpecificParameter;

template<class Weight>
class PopulationAlgorithm_: public algorithm::AlgorithmInterface<Weight> {
public:
	/**
	 * An algorithm should export its parameter type
	 */
	typedef parameters::PopulistParameter Parameter;

	/**
	 * Create a PopulistAlgorithm with settings defined in a PopulistParameter
	 * @param par_populist the settings of the algorithm
	 */
	PopulationAlgorithm_(const parameters::PopulistParameter& par_populist);

	/**
	 * copy constructor
	 * @param algorithm the rhs algorithm
	 */
	PopulationAlgorithm_(const PopulationAlgorithm_<Weight>& algorithm);

	/**
	 * virtual destructor
	 */
	virtual ~PopulationAlgorithm_();

	/**
	 * Configure the Algorithm
	 * @param simParam the Simulation Parameters
	 */
	virtual void configure(const SimulationRunParameter& simParam);

	/**
	 * Evolve the node state.
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 * @param typeVector Vector of the NodeTypes of the precursors
	 */
	virtual void evolveNodeState(const std::vector<MPILib::Rate>& nodeVector,
			const std::vector<Weight>& weightVector, Time time,
			const std::vector<MPILib::NodeType>& typeVector);

	/**
	 * prepare the Evolve method
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param weightVector Vector of the NodeTypes of the precursors
	 */
        virtual void prepareEvolve(const std::vector<Rate>& nodeVector, //!
			const std::vector<Weight>& weightVector,
			const std::vector<NodeType>& typeVector);

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

	/**
	 * Stores the algorithm state in a Algorithm Grid
	 * @return The state of the algorithm
	 */
	virtual algorithm::AlgorithmGrid getGrid() const;

	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual PopulationAlgorithm_<Weight>* clone() const {
		return new PopulationAlgorithm_<Weight>(*this);
	}

	/**
	 * Give the potential that corresponds to a bin number at a specific moment
	 * @param index The index of the bin
	 * @return The Potential of the index bin
	 */
	Potential BinToCurrentPotential(Index index) const;

	/**
	 * Give the bin number that momentarily corresponds to a potential
	 * @param v The Potential
	 * @return The index which correspond to the potential
	 */
	Index CurrentPotentialToBin(Potential v) const;

private:

	/**
	 * Embed the initial grid in the local grid
	 */
	void Embed();

	/**
	 * the PopulationParameter
	 */
	parameters::PopulationParameter _parameter_population;
	/**
	 * The PopulistSpecificParameter
	 */
	parameters::PopulistSpecificParameter _parameter_specific;
	/**
	 * The Algorithm grid
	 */
	algorithm::AlgorithmGrid _grid;
	/**
	 * The PopulationGridController
	 */
	PopulationGridController<Weight> _controller_grid;
	/**
	 * The current time
	 */
	Time _current_time = 0.0;
	/**
	 * The current rate
	 */
	Rate _current_rate = 0.0;

}
;
// end of PopulationAlgorithm

// default algorithm is with PopulistConnection
typedef PopulationAlgorithm_<DelayedConnection> PopulationAlgorithm;

typedef algorithm::RateAlgorithm<DelayedConnection> Pop_RateAlgorithm;
typedef RateFunctor<DelayedConnection> Pop_RateFunctor;
typedef MPINode<DelayedConnection, utilities::CircularDistribution> Pop_DynamicNode;
typedef MPINetwork<DelayedConnection, utilities::CircularDistribution> Pop_Network;

} /* namespace populist */
} /* namespace MPILib */

#endif // include guard MPILIB_POPULIST_POPOULATIONALGORITHM_HPP_
