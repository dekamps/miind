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
namespace MPILib {

template<class Weight, class NodeDistribution>
MPINode<Weight, NodeDistribution>::MPINode(
		const AlgorithmInterface<Weight>& algorithm, NodeType nodeType,
		NodeId nodeId,
		const boost::shared_ptr<NodeDistribution>& nodeDistribution,
		const boost::shared_ptr<std::map<NodeId, MPINode> >& localNode) :
		_algorithm(algorithm.Clone()), _nodeType(nodeType), _nodeId(nodeId), _pLocalNodes(
				localNode), _pNodeDistribution(nodeDistribution) {

}

template<class Weight, class NodeDistribution>
MPINode<Weight, NodeDistribution>::~MPINode() {
}

template<class Weight, class NodeDistribution>
Time MPINode<Weight, NodeDistribution>::Evolve(Time time) {

	receiveData();

	waitAll();
	while (_algorithm->getCurrentTime() < time) {
		++_number_iterations;

		_algorithm->EvolveNodeState(_precursorActivity, _weights, time);

	}

	// update state
	this->setActivity(_algorithm->getCurrentRate());

	sendOwnActivity();

	return _algorithm->getCurrentTime();
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::ConfigureSimulationRun(
		const DynamicLib::SimulationRunParameter& simParam) {
	_maximum_iterations = simParam.MaximumNumberIterations();
	_algorithm->Configure(simParam);

	// Add this line or other nodes will not get a proper input at the first simulation step!
	this->setActivity(_algorithm->getCurrentRate());

	_p_handler = auto_ptr<DynamicLib::AbstractReportHandler>(
			simParam.Handler().Clone());

	// by this time, the Id of a Node should be known
	// this can't be handled by the constructor because it is an implementation (i.e. a network)  property
	_info._id = NetLib::ConvertToNodeId(_nodeId);
	_p_handler->InitializeHandler(_info);

}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::addPrecursor(NodeId nodeId,
		const Weight& weight) {
	_precursors.push_back(nodeId);
	_weights.push_back(weight);
	//make sure that _precursorStates is big enough to store the data
	_precursorActivity.resize(_precursors.size());
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::addSuccessor(NodeId nodeId) {
	_successors.push_back(nodeId);
}

//template<class Weight, class NodeDistribution>
//DynamicLib::NodeState MPINode<Weight, NodeDistribution>::getState() const {
//	return _state;
//}
//
//template<class Weight, class NodeDistribution>
//void MPINode<Weight, NodeDistribution>::setState(DynamicLib::NodeState state) {
//	_state = state;
//}

template<class Weight, class NodeDistribution>
ActivityType MPINode<Weight, NodeDistribution>::getActivity() const {
	return _activity;
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::setActivity(ActivityType activity) {
	_activity = activity;
}

template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::waitAll() {
	mpi::wait_all(_mpiStatus.begin(), _mpiStatus.end());
	_mpiStatus.clear();

}
template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::receiveData() {

	std::vector<NodeId>::iterator it;
	int i = 0;
	for (it = _precursors.begin(); it != _precursors.end(); it++, i++) {
		mpi::communicator world;
		//do not send the data if the node is local!
		if (_pNodeDistribution->isLocalNode(*it)) {
			_precursorActivity[i] =
					_pLocalNodes->find(*it)->second.getActivity();

		} else {
			_mpiStatus.push_back(
					world.irecv(
							_pNodeDistribution->getResponsibleProcessor(*it),
							_pNodeDistribution->getRank(),
							_precursorActivity[i]));
		}
	}
}
template<class Weight, class NodeDistribution>
void MPINode<Weight, NodeDistribution>::sendOwnActivity() {

	std::vector<NodeId>::iterator it;
	for (it = _successors.begin(); it != _successors.end(); it++) {
		mpi::communicator world;
		//do not send the data if the node is local!
		if (!_pNodeDistribution->isLocalNode(*it)) {
			_mpiStatus.push_back(
					world.isend(
							_pNodeDistribution->getResponsibleProcessor(*it),
							*it, _activity));
		}
	}

}

template<class Weight, class NodeDistribution>
std::string MPINode<Weight, NodeDistribution>::reportAll(
		DynamicLib::ReportType type) const {

	string string_return("");

	std::vector<DynamicLib::ReportValue> vec_values;

	if (type == DynamicLib::RATE || type == DynamicLib::STATE) {
		DynamicLib::Report report(_algorithm->getCurrentTime(),
				DynamicLib::Rate(this->getActivity()),
				NetLib::NodeId(this->_nodeId),
				DynamicLib::NodeState(std::vector<double>(_activity)),
				_algorithm->Grid(), string_return, type, vec_values);

		_p_handler->WriteReport(report);
	}

	if (type == DynamicLib::UPDATE)
		_p_handler->Update();

	return string_return;
}

} //end namespace MPILib

#endif /* CODE_MPILIB_MPINODE_HPP_ */
