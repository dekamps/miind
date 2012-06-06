/*
 * MPINetwork.cpp
 *
 *  Created on: 25.05.2012
 *      Author: david
 */

#include <MPILib/include/MPINetwork.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <sstream>

namespace mpi = boost::mpi;
using namespace MPILib;

MPINetwork::MPINetwork():_pNodeDistribution(new utilities::CircularDistribution), _pLocalNodes(new std::map<NodeId, MPINode>) {


	if (_pNodeDistribution->isMaster()) {
		_maxNodeId = 0;
	}

	//set to one as no node should exist at this time point

}

MPINetwork::~MPINetwork() {

}

int MPINetwork::AddNode(const AlgorithmInterface& alg, NodeType nodeType) {

	int tempNodeId = getMaxNodeId();
	if (_pNodeDistribution->isLocalNode(tempNodeId)) {
		MPINode node = MPINode(alg, nodeType, tempNodeId, _pNodeDistribution, _pLocalNodes);
		_pLocalNodes->insert(std::make_pair(tempNodeId, node));
	}
	//increment the max NodeId to make sure that it is not assigned twice.
	incrementMaxNodeId();
	return tempNodeId;
}

void MPINetwork::MakeFirstInputOfSecond(NodeId first, NodeId second,
		const WeightType& weight) {

	if (_pNodeDistribution->isLocalNode(first)) {
		if (_pLocalNodes->count(first) > 0) {
			_pLocalNodes->find(first)->second.addSuccessor(second);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << first << "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}
	if (_pNodeDistribution->isLocalNode(second)) {
		if (_pLocalNodes->count(second) > 0) {
			_pLocalNodes->find(second)->second.addPrecursor(first, weight);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << second << "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}


}

void MPINetwork::ConfigureSimulation(const SimulationRunParameter& simParam) {
	//TODO implement this


}

//! Envolve the network
void MPINetwork::Evolve() {

	for (std::map<NodeId, MPINode>::iterator it = _pLocalNodes->begin();
			it != _pLocalNodes->end(); it++) {
		std::cout << "processorID:\t" << _pNodeDistribution->getRank();
		//FIXME change to better time
		it->second.Evolve(1);
		std::cout << std::endl;
		//TODO call evolve on the nodes
	}


}

int MPINetwork::getMaxNodeId() {

	mpi::communicator world;
	mpi::broadcast(world, _maxNodeId, 0);
	return _maxNodeId;
}

void MPINetwork::incrementMaxNodeId() {
	if (_pNodeDistribution->isMaster()) {
		_maxNodeId++;
	}
}
