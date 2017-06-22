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

#ifndef MPILIB_ALGORITHMS_DELAYASSEMBLYALGORITHM_HPP_
#define MPILIB_ALGORITHMS_DELAYASSEMBLYALGORITHM_HPP_

#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/include/DelayAssemblyParameter.hpp>

#include <MPILib/include/AlgorithmInterface.hpp>

namespace MPILib {

  template <class WeightType>
class DelayAssemblyAlgorithm: public AlgorithmInterface<WeightType> {
public:
 
	DelayAssemblyAlgorithm();

	DelayAssemblyAlgorithm(const DelayAssemblyParameter&);

	virtual ~DelayAssemblyAlgorithm();

	/**
	 * Cloning operation, to provide each DynamicNode with its own
	 * Algorithm instance. Clients use the naked pointer at their own risk.
	 */
	virtual DelayAssemblyAlgorithm* clone() const;

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
			const std::vector<WeightType>& weightVector, Time time);

	/**
	 * The current timepoint
	 * @return The current time point
	 */
	virtual Time getCurrentTime() const { return _t_current; }

	/**
	 * The calculated rate of the node
	 * @return The rate of the node
	 */
	virtual Rate getCurrentRate() const {return _r_current; }

	virtual AlgorithmGrid getGrid(NodeId) const;

private:

	double innerProduct(const std::vector<Rate>& nodeVector,
			const std::vector<double>& weightVector);

	std::vector<double> getInitialState() const;

	DelayAssemblyParameter _par;
	MPILib::Time           _t_current;
	MPILib::Rate           _r_current;
	MPILib::Time           _last_activation;
	MPILib::Rate           _change_factor;
};

} /* namespace MPILib */
#endif /* MPILIB_ALGORITHMS_DELAYASSEMBLYALGORITHM_HPP_ */
