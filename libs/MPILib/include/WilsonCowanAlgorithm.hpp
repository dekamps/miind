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

#ifndef MPILIB_ALGORITHMS_WILSONCOWANALGORITHM_HPP_
#define MPILIB_ALGORITHMS_WILSONCOWANALGORITHM_HPP_

#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/WilsonCowanParameter.hpp>

#include <MPILib/include/AlgorithmInterface.hpp>

namespace MPILib {

/*! \page wilson_cowan The Wilson-Cowan Algorithm
 *  \section wilson_cowan_introduction The Wilson-Cowan Algorithm
 *  The Wilson-Cowan Algorithm is probably the best known neural mass model.
 */

/**
 * @brief The background of this algorithm is described on page \ref wilson_cowan. An example of a fully functional programming
 * using this algorithm is also presented there. Here we present the documentation required by C++
 * clients of this algorithm.
 *
 * This algorithm is defined for usage in MPINetwork. This describes network of nodes connected by link which are
 * described by a single number which internally are represented by a double.
 *
 */

class WilsonCowanAlgorithm: public AlgorithmInterface<double> {
public:
 
	WilsonCowanAlgorithm();

	WilsonCowanAlgorithm(const WilsonCowanParameter&);

	virtual ~WilsonCowanAlgorithm();

	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual WilsonCowanAlgorithm* clone() const;

	/**
	 * Configure the Algorithm
	 * @param simParam
	 */
	virtual void configure(const SimulationRunParameter& simParam);

	/**
	 * Evolve the node state
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param time Time point of the algorithm
	 */
	virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
			const std::vector<double>& weightVector, Time time);

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

	virtual AlgorithmGrid getGrid(NodeId, bool b_state = true) const;

private:

	double innerProduct(const std::vector<Rate>& nodeVector,
			const std::vector<double>& weightVector);

	std::vector<double> getInitialState() const;

	WilsonCowanParameter _parameter;

	NumtoolsLib::DVIntegrator<WilsonCowanParameter> _integrator;

};

} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMS_WILSONCOWANALGORITHM_HPP_ */
