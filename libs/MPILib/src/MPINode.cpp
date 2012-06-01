/*
 * MPINode.cpp
 *
 *  Created on: 29.05.2012
 *      Author: David Sichau
 */

#include <MPILib/include/MPINode.hpp>
#include <iostream>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>

namespace mpi = boost::mpi;
using namespace MPILib;

MPINode::MPINode(const Algorithm& algorithm, NodeType nodeType, NodeId nodeId,
		const boost::shared_ptr<utilities::NodeDistributionInterface>& nodeDistribution) :
		_algorithm(algorithm), _nodeType(nodeType), _nodeId(nodeId), _nodeDistribution(
				nodeDistribution) {

}
;

MPINode::~MPINode() {
}
;

Time MPINode::Evolve(Time time) {
	waitAll();

	std::cout << " # \t NodeId: " << _nodeId << "\t precursor size: "
			<< _precursors[0] << "\t successors size: " << _successors[0]
			<< " # ";

	receiveData();
	sendOwnState();

	//FIXME Implement this stub
	return 0;
}

bool MPINode::ConfigureSimulationRun(const SimulationRunParameter& simParam) {
	//FIXME Implement this stub
	return true;
}

void MPINode::addPrecursor(NodeId nodeId, const WeightType& weight) {
	_precursors.push_back(nodeId);
	_weights.push_back(weight);
	//make sure that _precursorStates is big enough to store the data
	_precursorStates.resize(_precursors.size());
}

void MPINode::addSuccessor(NodeId nodeId, const WeightType& weight) {
	_successors.push_back(nodeId);
}

NodeState MPINode::getState() const {
	return _state;
}

void MPINode::setState(NodeState state) {
	_state = state;
}

void ::MPINode::waitAll() {
	mpi::wait_all(_mpiStatus.begin(), _mpiStatus.end());
	_mpiStatus.clear();
}

void MPINode::receiveData() {

	std::vector<NodeId>::iterator it;
	int i = 0;
	for (it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		mpi::communicator world;
		_mpiStatus.push_back(
				world.irecv(_nodeDistribution->getResponsibleProcessor(*it),
						*it, _precursorStates[i]));
	}
}

void MPINode::sendOwnState() {

	std::vector<NodeId>::iterator it;
	for (it = _successors.begin(); it != _successors.end(); it++) {
		mpi::communicator world;
		_mpiStatus.push_back(
				world.isend(_nodeDistribution->getResponsibleProcessor(*it),
						*it, _state));
	}

}
