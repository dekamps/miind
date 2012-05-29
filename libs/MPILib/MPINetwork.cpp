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

int MPINetwork::AddNode(const Algorithm& alg, NodeType nodeType) {

	int tempNodeId = getMaxNodeId();
	if (isLocalNode(tempNodeId)) {
		MPINode node = MPINode(alg, nodeType, tempNodeId);
		_localNodes.insert(std::make_pair(tempNodeId, node));
	}
	//increment the max NodeId to make sure that it is not assigned twice.
	//TODO why here? It does not work correct in the if condition above.
	incrementMaxNodeId();
	return tempNodeId;
}

bool MPINetwork::MakeFirstInputOfSecond(NodeId first, NodeId second,
		WeightType& weight) {

	assert(first!=second);

	if (isLocalNode(first)) {
		if (_localNodes.count(first) > 0) {
			_localNodes.find(first)->second.addSuccessor(second, weight);
		} else {
			return false;
		}
	}
	if (isLocalNode(second)) {
		if (_localNodes.count(second) > 0) {
			_localNodes.find(second)->second.addPrecursor(first, weight);
		} else {
			return false;
		}
	}

	return true;

}

bool MPINetwork::ConfigureSimulation(const SimulationRunParameter& simParam) {
	//TODO implement this

	//FIXME change to correct return
	return true;
}

//! Envolve the network
bool MPINetwork::Evolve() {

	for (std::map<NodeId, MPINode>::iterator it = _localNodes.begin();
			it != _localNodes.end(); it++) {
		std::cout << "processorID:\t" << _processorId;
		//FIXME change to better time
		it->second.Evolve(1);
		std::cout << std::endl;
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
