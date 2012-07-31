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

#ifndef MPILIB_ALGORITHMS_DELAYALGORITHM_HPP_
#define MPILIB_ALGORITHMS_DELAYALGORITHM_HPP_

#include <deque>
#include <MPILib/include/TypeDefinitions.hpp>

#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include <MPILib/include/SimulationRunParameter.hpp>

namespace MPILib {
namespace algorithm {

/**
 * @brief This algorithm is effectively a pipeline with a preselected delay.
 *
 * In some simulations connections must be implemented with time delays. If that needs to be done with
 * high precision, create a node, configure it with a DelayAlgorithm, connected the output to be delayed
 * to this node and connect the output of this node to the node specified by the original connection. At the
 * moment this is the only way to implement delays. A less precise effect can be achieved with Wilson-Cowan algorithms.
 * For large-scale simulations this solution may not be sustainable. Please provide feedback if this is the case.
 *
 */
template<class Weight>
class DelayAlgorithm: public AlgorithmInterface<Weight> {
public:

	/**
	 * Create algorithm with a delay time
	 * @param t_delay The delay time
	 */
	DelayAlgorithm(Time t_delay);

	/**
	 * virtual destructor
	 */
	virtual ~DelayAlgorithm();

	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual DelayAlgorithm<Weight>* clone() const;
	/**
	 * Configure the Algorithm
	 * @param simParam The simulation parameter
	 */
	virtual void configure(const SimulationRunParameter& simParam);

	/**
	 * Evolve the node state. Overwrite this method if your algorithm does not
	 * need to know the NodeTypes.
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<Weight>& weightVector, Time time);

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
	virtual AlgorithmGrid getGrid() const;

private:

	Rate CalculateDelayedRate();

	Rate Interpolate() const;

	typedef std::pair<Rate, Time> rate_time_pair;

	Time _t_current;
	Time _t_delay;
	Rate _rate_current;

	std::deque<rate_time_pair> _queue;

};

} /* end namespace algorithm */
} /* end namespace MPILib */

#endif  //include guard MPILIB_ALGORITHMS_DELAYALGORITHM_HPP_
