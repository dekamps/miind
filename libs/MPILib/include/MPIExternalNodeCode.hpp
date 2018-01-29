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

#ifndef CODE_MPILIB_MPIEXTERNALNODE_HPP_
#define CODE_MPILIB_MPIEXTERNALNODE_HPP_

#include <MPILib/include/MPINode.hpp>
#include <iostream>
#include <MPILib/include/BasicDefinitions.hpp>
#include <MPILib/include/RateAlgorithmCode.hpp>
#include <MPILib/include/utilities/Exception.hpp>
#include <MPILib/include/utilities/Log.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>
namespace MPILib {

template<class Weight, class NodeDistribution>
MPIExternalNode<Weight, NodeDistribution>::MPIExternalNode(
		const NodeDistribution& nodeDistribution,
		const std::map<NodeId, MPINode<Weight, NodeDistribution>>& localNode,
		const std::string& name) : MPINode<Weight, NodeDistribution>(RateAlgorithm<Weight>(0), NEUTRAL, 0, nodeDistribution, localNode, name){

}

template<class Weight, class NodeDistribution>
MPIExternalNode<Weight, NodeDistribution>::~MPIExternalNode() {
}

template<class Weight, class NodeDistribution>
void MPIExternalNode<Weight, NodeDistribution>::setActivities(std::vector<ActivityType> activities) {

  _activities = activities;
	sendOwnActivity();
}

template<class Weight, class NodeDistribution>
std::vector<ActivityType> MPIExternalNode<Weight, NodeDistribution>::getPrecursorActivity() {
  receiveData();
  return this->_precursorActivity;
}

template<class Weight, class NodeDistribution>
Time MPIExternalNode<Weight, NodeDistribution>::evolve(Time time) {

	++this->_number_iterations;

	return time;
}

template<class Weight, class NodeDistribution>
void MPIExternalNode<Weight, NodeDistribution>::prepareEvolve() {
  // No algorithm to prepare so do nothing
}

template<class Weight, class NodeDistribution>
void MPIExternalNode<Weight, NodeDistribution>::configureSimulationRun(
		const SimulationRunParameter& simParam) {

	this->_maximum_iterations = simParam.getMaximumNumberIterations();

	// No handler is initialised for the external node
}

template<class Weight, class NodeDistribution>
void MPIExternalNode<Weight, NodeDistribution>::addPrecursor(NodeId nodeId,
		const Weight& weight, NodeType nodeType) {
	this->_precursors.push_back(nodeId);
	this->_precursorTypes.push_back(nodeType);
	this->_weights.push_back(weight);
	//make sure that _precursorStates is big enough to store the data
	this->_precursorActivity.resize(this->_precursors.size());
}

template<class Weight, class NodeDistribution>
void MPIExternalNode<Weight, NodeDistribution>::addSuccessor(NodeId nodeId) {
	this->_successors.push_back(nodeId);
  _activities.resize(this->_successors.size());
}

template<class Weight, class NodeDistribution>
void MPIExternalNode<Weight, NodeDistribution>::receiveData() {
	int i = 0;

	for (auto it = this->_precursors.begin(); it != this->_precursors.end(); it++, i++) {
		//do not send the data if the node is local!
		if (this->_rNodeDistribution.isLocalNode(*it)) {
			this->_precursorActivity[i] =
					this->_rLocalNodes.find(*it)->second.getActivity();

		} else {
			utilities::MPIProxy().irecv(this->_rNodeDistribution.getResponsibleProcessor(*it), *it,
					this->_precursorActivity[i]);
		}
	}
}

template<class Weight, class NodeDistribution>
void MPIExternalNode<Weight, NodeDistribution>::sendOwnActivity() {
  int i = 0;
	for (auto& it : this->_successors) {
		//do not send the data if the node is local!
		if (!this->_rNodeDistribution.isLocalNode(it)) {
			utilities::MPIProxy().isend(this->_rNodeDistribution.getResponsibleProcessor(it),
					this->_nodeId, _activities[i]);
		}
    i++;
	}
}

template<class Weight, class NodeDistribution>
void MPIExternalNode<Weight, NodeDistribution>::reportAll(
		report::ReportType type) const {
	// No reporting from the external node
}

template<class Weight, class NodeDistribution>
void MPIExternalNode<Weight, NodeDistribution>::clearSimulation() {
  // no reporting from the external node
}

}
//end namespace MPILib

#endif /* CODE_MPILIB_MPIEXTERNALNODE_HPP_ */
