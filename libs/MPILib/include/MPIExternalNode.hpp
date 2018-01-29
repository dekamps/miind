// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
//            2018 Hugh Osborne
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

#ifndef MPILIB_MPIEXTERNALNODE_HPP_
#define MPILIB_MPIEXTERNALNODE_HPP_

#include <vector>
#include <map>
#include <memory>

#include <MPILib/include/AlgorithmInterface.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <MPILib/include/NodeType.hpp>

#include <MPILib/include/TypeDefinitions.hpp>

namespace MPILib {

template<class Weight, class NodeDistribution>
class MPIExternalNode : public MPINode<Weight, NodeDistribution> {
public:
	/**
	 * Constructor
	 * @param nodeDistribution The Node Distribution.
	 * @param localNode The local nodes of this processor
	 */
	explicit MPIExternalNode(
			const NodeDistribution& nodeDistribution,
			const std::map<NodeId, MPINode<Weight, NodeDistribution>>& localNode,
			const std::string& name = ""
			);

	/**
	 * Destructor
	 */
	virtual ~MPIExternalNode();

	/**
	 * Evolve this algorithm over a time
	 * @param time Time until the algorithm should evolve
	 * @return Time the algorithm have evolved, which may be slightly different, due to rounding errors.
	 */
	Time evolve(Time time);

	/**
	 * Called before each evolve call during each evolve. Can
	 * be used to prepare the input for the evolve method.
	 */
	void prepareEvolve();

	/**
	 * Configure the Node with the Simulation Parameters
	 * @param simParam Simulation Parameters
	 */
	void configureSimulationRun(const SimulationRunParameter& simParam);

  /**
	 * Add a precursor to the current node
	 * @param nodeId NodeId the id of the precursor
	 * @param weight the weight of the connection
	 * @param nodeType the nodeType of the precursor
	 */
	void addPrecursor(NodeId nodeId, const Weight& weight, NodeType nodeType);

	/**
	 * Add a successor to the current node
	 * @param nodeId NodeId the id of the successor
	 */
	void addSuccessor(NodeId nodeId);

	/**
	 * Receive the new data from the precursor nodes
	 */
	void receiveData();

	/**
	 * Send the own state to the successors.
	 */
	void sendOwnActivity();

	/**
	 * Report the node state
	 * @param type The type of Report
	 */
	void reportAll(report::ReportType type) const;

	/**
	 * finishes the simulation.
	 */
	void clearSimulation();

  void setActivities(std::vector<ActivityType> activities);

  std::vector<ActivityType> getPrecursorActivity();

protected:

  std::vector<ActivityType> _activities;

};

} //end namespace

#endif /* MPILIB_MPIEXTERNALNODE_HPP_ */
