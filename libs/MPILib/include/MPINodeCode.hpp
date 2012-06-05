/*
 * MPINode.cpp
 *
 *  Created on: 29.05.2012
 *      Author: David Sichau
 */

#ifndef CODE_MPILIB_MPINODE_HPP_
#define CODE_MPILIB_MPINODE_HPP_

#include <MPILib/include/MPINode.hpp>
#include <iostream>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>

namespace mpi = boost::mpi;
namespace MPILib{

template<class Weight>
MPINode<Weight>::MPINode(const AlgorithmInterface<Weight>& algorithm, NodeType nodeType,
		NodeId nodeId,
		const boost::shared_ptr<utilities::NodeDistributionInterface>& nodeDistribution,
		const std::map<NodeId, MPINode>& localNode) :
		_algorithm(algorithm.Clone()), _nodeType(nodeType), _nodeId(nodeId), _nodeDistribution(
				nodeDistribution), _refLocalNodes(localNode) {

}
;
template<class Weight>
MPINode<Weight>::~MPINode() {
}
;
template<class Weight>
Time MPINode<Weight>::Evolve(Time time) {

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
				<< _precursorStates[i] << " # " << std::endl;

	}
	//FIXME Implement this stub
	return 0;
}
template<class Weight>
void MPINode<Weight>::ConfigureSimulationRun(
		const SimulationRunParameter& simParam) {
	//FIXME Implement this stub
}
template<class Weight>
void MPINode<Weight>::addPrecursor(NodeId nodeId, const Weight& weight) {
	_precursors.push_back(nodeId);
	_weights.push_back(weight);
	//make sure that _precursorStates is big enough to store the data
	_precursorStates.resize(_precursors.size());
}
template<class Weight>
void MPINode<Weight>::addSuccessor(NodeId nodeId) {
	_successors.push_back(nodeId);
}
template<class Weight>
NodeState MPINode<Weight>::getState() const {
	return _state;
}
template<class Weight>
void MPINode<Weight>::setState(NodeState state) {
	_state = state;
}
template<class Weight>
void MPINode<Weight>::waitAll() {
	mpi::wait_all(_mpiStatus.begin(), _mpiStatus.end());
	_mpiStatus.clear();

}
template<class Weight>
void MPINode<Weight>::receiveData() {

	std::vector<NodeId>::iterator it;
	int i = 0;
	for (it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		mpi::communicator world;
		//do not send the data if the node is local!
		if (_nodeDistribution->isLocalNode(*it)) {
			_precursorStates[i] = _refLocalNodes.find(*it)->second.getState();

		} else {
			_mpiStatus.push_back(
					world.irecv(_nodeDistribution->getResponsibleProcessor(*it),
							_nodeDistribution->getRank(), _precursorStates[i]));
		}
	}
}
template<class Weight>
void MPINode<Weight>::sendOwnState() {

	std::vector<NodeId>::iterator it;
	for (it = _successors.begin(); it != _successors.end(); it++) {
		mpi::communicator world;
		//do not send the data if the node is local!
		if (!_nodeDistribution->isLocalNode(*it)) {
			_mpiStatus.push_back(
					world.isend(_nodeDistribution->getResponsibleProcessor(*it),
							*it, _state));
		}
	}

}
}//end namespace MPILib

#endif /* CODE_MPILIB_MPINODE_HPP_ */
