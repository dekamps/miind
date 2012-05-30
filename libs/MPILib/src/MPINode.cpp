/*
 * MPINode.cpp
 *
 *  Created on: 29.05.2012
 *      Author: david
 */

#include <MPILib/include/MPINode.hpp>
#include <iostream>

#include <boost/mpi/communicator.hpp>

namespace mpi = boost::mpi;
using namespace MPILib;


MPINode::MPINode(const Algorithm& algorithm, NodeType nodeType, NodeId nodeId) :
		_algorithm(algorithm), _nodeType(nodeType), _nodeId(nodeId) {

	mpi::communicator world;

	_processorId = world.rank();
	_totalProcessors = world.size();
}
;

MPINode::~MPINode() {
}
;

Time MPINode::Evolve(Time time) {
	std::cout << " # \t NodeId: " << _nodeId << "\t precursor size: "
			<< _precursors[0].first << "\t successors size: "
			<< _successors[0].first << " # ";
	//FIXME Implement this stub
	return 0;
}

bool MPINode::ConfigureSimulationRun(const SimulationRunParameter& simParam) {
	//FIXME Implement this stub
	return true;
}

void MPINode::addPrecursor(NodeId nodeId, const WeightType& weight) {
	_precursors.push_back(std::make_pair(nodeId, weight));
}

void MPINode::addSuccessor(NodeId nodeId, const WeightType& weight) {
	_successors.push_back(std::make_pair(nodeId, weight));
}

NodeState MPINode::getState() const {
	return _state;
}

void MPINode::setState(NodeState state) {
	_state = state;
}


void MPINode::receiveData(){

}


void MPINode::sendOwnState(){

}
