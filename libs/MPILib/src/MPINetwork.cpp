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

MPINetwork::MPINetwork():_mpiDistribution(new utilities::CircularDistribution) {


	if (_mpiDistribution->isMaster()) {
		_maxNodeId = 0;
	}

	//set to one as no node should exist at this time point

}

MPINetwork::~MPINetwork() {

}

int MPINetwork::AddNode(const Algorithm& alg, NodeType nodeType) {

	int tempNodeId = getMaxNodeId();
	if (_mpiDistribution->isLocalNode(tempNodeId)) {
		MPINode node = MPINode(alg, nodeType, tempNodeId);
		_localNodes.insert(std::make_pair(tempNodeId, node));
	}
	//increment the max NodeId to make sure that it is not assigned twice.
	incrementMaxNodeId();
	return tempNodeId;
}

void MPINetwork::MakeFirstInputOfSecond(NodeId first, NodeId second,
		const WeightType& weight) {

	if (_mpiDistribution->isLocalNode(first)) {
		if (_localNodes.count(first) > 0) {
			_localNodes.find(first)->second.addSuccessor(second, weight);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << first << "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}
	if (_mpiDistribution->isLocalNode(second)) {
		if (_localNodes.count(second) > 0) {
			_localNodes.find(second)->second.addPrecursor(first, weight);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << second << "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}


}

bool MPINetwork::ConfigureSimulation(const SimulationRunParameter& simParam) {
	//TODO implement this

	//FIXME change to correct return
	return true;
}

//! Envolve the network
void MPINetwork::Evolve() {

	for (std::map<NodeId, MPINode>::iterator it = _localNodes.begin();
			it != _localNodes.end(); it++) {
		std::cout << "processorID:\t" << _mpiDistribution->getRank();
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
	if (_mpiDistribution->isMaster()) {
		_maxNodeId++;
	}
}
