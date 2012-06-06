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

MPINode::MPINode(const AlgorithmInterface& algorithm, NodeType nodeType, NodeId nodeId,
		const boost::shared_ptr<utilities::NodeDistributionInterface>& nodeDistribution,
		const boost::shared_ptr<std::map<NodeId, MPINode> > localNode) :
		_algorithm(algorithm.Clone()), _nodeType(nodeType), _nodeId(nodeId), _pNodeDistribution(
				nodeDistribution), _pLocalNodes(localNode) {

}
;

MPINode::~MPINode() {
}
;

Time MPINode::Evolve(Time time) {

	_algorithm->EvolveNodeState(_precursorStates, _weights, time);


	_state = _algorithm->getCurrentRate();

	std::cout << "#\t NodeId: " << _nodeId << "\t#precursors: "
			<< _precursors.size() << "\t#successors: " << _successors.size()
			<< "\t state: " << _state << std::endl;

	receiveData();
	sendOwnState();
	waitAll();

	for (int i = 0; i < _precursorStates.size(); i++) {
		std::cout << " # \t NodeId: " << _nodeId << "\t state of precursor: "
				<< _precursorStates[i] << " # "<<std::endl;

	}
	//FIXME Implement this stub
	return 0;
}

void MPINode::ConfigureSimulationRun(const SimulationRunParameter& simParam) {
	//FIXME Implement this stub
}

void MPINode::addPrecursor(NodeId nodeId, const WeightType& weight) {
	_precursors.push_back(nodeId);
	_weights.push_back(weight);
	//make sure that _precursorStates is big enough to store the data
	_precursorStates.resize(_precursors.size());
}

void MPINode::addSuccessor(NodeId nodeId) {
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
		//do not send the data if the node is local!
		if (_pNodeDistribution->isLocalNode(*it)) {
			_precursorStates[i] = _pLocalNodes->find(*it)->second.getState();

		} else {
			_mpiStatus.push_back(
					world.irecv(_pNodeDistribution->getResponsibleProcessor(*it),
							_pNodeDistribution->getRank(), _precursorStates[i]));
		}
	}
}

void MPINode::sendOwnState() {

	std::vector<NodeId>::iterator it;
	for (it = _successors.begin(); it != _successors.end(); it++) {
		mpi::communicator world;
		//do not send the data if the node is local!
		if (!_pNodeDistribution->isLocalNode(*it)) {
			_mpiStatus.push_back(
					world.isend(_pNodeDistribution->getResponsibleProcessor(*it),
							*it, _state));
		}
	}

}
