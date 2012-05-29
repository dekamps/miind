/*
 * MPINetwork.cpp
 *
 *  Created on: 25.05.2012
 *      Author: david
 */

#include "MPINetwork.hpp"
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

namespace mpi = boost::mpi;

MPINetwork::MPINetwork() {

	mpi::communicator world;

	_processorId = world.rank();
	_totalProcessors = world.size();

	if (isMaster()) {
		_maxNodeId = 0;
	}

	//set to one as no node should exist at this time point

}

MPINetwork::~MPINetwork() {

}

void MPINetwork::AddNode(const Algorithm& alg, NodeType nodetype) {

	int tempNodeId = getMaxNodeId();
	if (isLocalNode(tempNodeId)) {
		//TODO make new node
		//FIXME push back the actual nodes
		_localNodes.push_back(tempNodeId);
	}
	//increment the max NodeId to make sure that it is not assigned twice.
	incrementMaxNodeId();
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
		std::cout << "processorID:\t" << _processorId << "\tNodeId:\t" << *it
				<< "\t" << std::endl;
		//TODO call evolve on the nodes
	}
	//FIXME change this
	return true;

}

bool MPINetwork::isLocalNode(NodeId nodeId) {
	return getResponsibleProcessor(nodeId) == _processorId;
}

int MPINetwork::getResponsibleProcessor(NodeId nodeId) {
	return nodeId % _totalProcessors;
}

bool MPINetwork::isMaster() {
	return _processorId == 0;
}

int MPINetwork::getMaxNodeId() {

	mpi::communicator world;
	mpi::broadcast(world, _maxNodeId, 0);
	return _maxNodeId;
}

void MPINetwork::incrementMaxNodeId() {
	if (isMaster()) {
		_maxNodeId++;
	}
}
