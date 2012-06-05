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

template <class WeightValue>
MPINetwork<WeightValue>::MPINetwork():_nodeDistribution(new utilities::CircularDistribution) {


	if (_nodeDistribution->isMaster()) {
		_maxNodeId = 0;
	}

	//set to one as no node should exist at this time point

}

template <class WeightValue>
MPINetwork<WeightValue>::~MPINetwork() {

}

template <class WeightValue>
int MPINetwork<WeightValue>::AddNode(const AlgorithmInterface& alg, NodeType nodeType) {

	int tempNodeId = getMaxNodeId();
	if (_nodeDistribution->isLocalNode(tempNodeId)) {
		D_Node node = D_Node(alg, nodeType, tempNodeId, _nodeDistribution, _localNodes);
		_localNodes.insert(std::make_pair(tempNodeId, node));
	}
	//increment the max NodeId to make sure that it is not assigned twice.
	incrementMaxNodeId();
	return tempNodeId;
}

template <class WeightValue>
void MPINetwork<WeightValue>::MakeFirstInputOfSecond(NodeId first, NodeId second,
		const WeightValue& weight) {

	if (_nodeDistribution->isLocalNode(first)) {
		if (_localNodes.count(first) > 0) {
			_localNodes.find(first)->second.addSuccessor(second);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << first << "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}
	if (_nodeDistribution->isLocalNode(second)) {
		if (_localNodes.count(second) > 0) {
			_localNodes.find(second)->second.addPrecursor(first, weight);
		} else {
			std::stringstream tempStream;
			tempStream << "the node " << second << "does not exist on this node";
			miind_parallel_fail(tempStream.str());
		}
	}


}

template <class WeightValue>
void MPINetwork<WeightValue>::ConfigureSimulation(const SimulationRunParameter& simParam) {
	//TODO implement this


}

//! Envolve the network
template <class WeightValue>
void MPINetwork<WeightValue>::Evolve() {

	for (std::map<NodeId, D_MPINode>::iterator it = _localNodes.begin();
			it != _localNodes.end(); it++) {
		std::cout << "processorID:\t" << _nodeDistribution->getRank();
		//FIXME change to better time
		it->second.Evolve(1);
		std::cout << std::endl;
		//TODO call evolve on the nodes
	}


}

template <class WeightValue>
int MPINetwork<WeightValue>::getMaxNodeId() {

	mpi::communicator world;
	mpi::broadcast(world, _maxNodeId, 0);
	return _maxNodeId;
}

template <class WeightValue>
void MPINetwork<WeightValue>::incrementMaxNodeId() {
	if (_nodeDistribution->isMaster()) {
		_maxNodeId++;
	}
}
