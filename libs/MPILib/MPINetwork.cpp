/*
 * MPINetwork.cpp
 *
 *  Created on: 25.05.2012
 *      Author: david
 */

#include "MPINetwork.hpp"
#include <boost/mpi/communicator.hpp>

namespace mpi = boost::mpi;

MPINetwork::MPINetwork() {

	mpi::communicator world;

	_processorId = world.rank();
	//set to one as no node should exist at this time point
	_maxNodeId = 0;

}

MPINetwork::~MPINetwork() {

}

NodeId MPINetwork::AddNode(const Algorithm& alg, NodeType nodetype) {
	//increase the maxNodeId by one to make sure that the node gets a new ID
	_maxNodeId++;
	if (isLocalNode(_maxNodeId)) {
		//TODO make new node
		//FIXME push back the actual nodes
		_localNodes.push_back(_maxNodeId);
	}
	return _maxNodeId;
}

bool MPINetwork::MakeFirstInputOfSecond(NodeId first, NodeId second,
		WeightType& weight) {

	assert(first!=second);

	//TODO connect the nodes

	//FIXME change to correct return
	return true;

}

bool MPINetwork::ConfigureSimulation(const SimulationRunParameter& simParam) {
	//TODO implement this

	//FIXME change to correct return
	return true;
}

//! Envolve the network
bool MPINetwork::Evolve() {

	for (std::vector<Node>::iterator it = _localNodes.begin();
			it != _localNodes.end(); it++) {
		//TODO call evolve on the nodes
	}
	//FIXME change this
	return true;

}

bool MPINetwork::isLocalNode(NodeId nodeId) {
	return getResponsibleProcessor(nodeId) == _processorId;
}

int MPINetwork::getResponsibleProcessor(NodeId nodeId) {
	return nodeId % _processorId;
}

